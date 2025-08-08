# app_stremlit/tests/test_cuda_sam2.py
import os
import pytest
import torch

# We only assert CUDA availability if user selected CUDA.
# Skip if no GPU present on CI to avoid false failures.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")
def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA should be available for this test to run"

def test_sam2_checkpoint_present_or_handled():
    """
    This test ensures that your code can instantiate the Segmenter either with a valid checkpoint
    or gracefully handle its absence (stub fallback).
    """
    from segmenter import Segmenter

    # Env or default path
    model_dir = os.environ.get("MODEL_DIR", "checkpoints")
    ckpt = os.environ.get("SAM2_CKPT", "sam2_hiera_l.pt")
    seg = Segmenter(device="cuda", model_dir=model_dir, sam2_checkpoint=ckpt)
    # If stub is used, predictor is still constructed; if real is available, checkpoint should be used.
    assert seg.langsam is not None
