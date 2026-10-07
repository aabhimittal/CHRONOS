import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from chronos.prefill import ChunkedPrefill, tiny_llama  # noqa: E402


@pytest.mark.parametrize("chunk", [1, 7, 32, 200])
def test_chunked_prefill_matches_unchunked(chunk):
    model = tiny_llama()
    ids = torch.randint(0, 512, (2, 150), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        ref = model(input_ids=ids).logits[:, -1]
    ex = ChunkedPrefill(model, input_ids=ids, chunk_tokens=chunk)
    while not ex.step(0.0):            # zero budget: exactly one chunk per slice
        pass
    assert ex.chunks == -(-150 // chunk)
    assert torch.allclose(ex.logits, ref, atol=1e-4)


def test_chunk_size_sets_slice_granularity():
    ex = ChunkedPrefill(tiny_llama(), input_ids=torch.zeros(1, 64, dtype=torch.long), chunk_tokens=16)
    ex.step(0.0)
    assert ex.pos == 16 and not ex.done
