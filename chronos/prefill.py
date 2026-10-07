"""Token-chunked prefill: preemption finer than a layer (requires torch + transformers).

The real-clock runtime showed that yielding only *between layers* cannot
honour a 10 ms slice once one layer of a batched VLM prefill takes longer
than that. Chunked prefill (Sarathi-Serve) splits the *prompt* instead: run
the first c tokens through all layers, keep the KV cache, run the next c
tokens, and so on. Attention over the cache makes the result identical to an
unchunked forward (tests check the logits), and the scheduler may yield after
any chunk, so the slice granularity is set by c, not by the model's depth.

Works with any Hugging Face decoder that accepts `past_key_values` and
`inputs_embeds`/`input_ids` (Llama, SmolLM, the text tower of SmolVLM). The
image encoder of a VLM runs once up front and is not chunked here.
"""
from __future__ import annotations

import time


class ChunkedPrefill:
    """Same interface as runtime.LayerSliced: `.step(budget) -> done`."""

    def __init__(self, model, input_ids=None, inputs_embeds=None, chunk_tokens=32):
        import torch
        self.torch, self.model, self.c = torch, model, chunk_tokens
        self.ids, self.emb = input_ids, inputs_embeds
        self.n = (input_ids if input_ids is not None else inputs_embeds).shape[1]
        self.pos, self.past, self.logits, self.chunks = 0, None, None, 0

    @property
    def done(self):
        return self.pos >= self.n

    def _run_chunk(self):
        sl = slice(self.pos, min(self.pos + self.c, self.n))
        kw = {"input_ids": self.ids[:, sl]} if self.ids is not None else {"inputs_embeds": self.emb[:, sl]}
        with self.torch.inference_mode():
            out = self.model(**kw, past_key_values=self.past, use_cache=True,
                             cache_position=self.torch.arange(sl.start, sl.stop))
        self.past, self.logits = out.past_key_values, out.logits[:, -1]
        self.pos, self.chunks = sl.stop, self.chunks + 1

    def step(self, budget: float) -> bool:
        t0 = time.perf_counter()
        while not self.done:
            self._run_chunk()
            if time.perf_counter() - t0 >= budget:
                break
        return self.done


def tiny_llama(layers=4, hidden=128, vocab=512, seed=0):
    """Randomly initialised Llama for tests and timing (no download)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 4,
                      num_hidden_layers=layers, num_attention_heads=max(hidden // 64, 1),
                      num_key_value_heads=max(hidden // 64, 1), max_position_embeddings=4096)
    return LlamaForCausalLM(cfg).eval()
