"""Model mechanics and typographic policy tests."""
import torch

from medfusion.models.components import RetrievalAugmentedDecoder
from medfusion.models.vlm import MedFusionVLM
from medfusion.training.losses import multitask_loss
from medfusion.utils import PROJECT_ROOT, load_config
from tools.hyphen_audit import audit_tree


def tiny_config():
    cfg = load_config()
    cfg["model"].update({"d_model": 32, "n_heads": 2, "text_layers": 1, "fusion_layers": 1, "decoder_layers": 1,
                         "ct_channels": [8, 16, 32], "wsi_channels": [8, 16, 32]})
    return cfg


def fake_batch(vocab=60, b=2):
    torch.manual_seed(0)
    labels = torch.randint(4, vocab, (b, 12))
    labels[:, :3] = 0
    return {
        "ct": torch.rand(b, 1, 128, 128), "wsi": torch.rand(b, 3, 64, 64), "wsi_present": torch.tensor([True, False]),
        "note_ids": torch.randint(4, vocab, (b, 20)), "findings_ids": torch.randint(4, vocab, (b, 16)),
        "decoder_input": torch.randint(4, vocab, (b, 12)), "labels": labels, "heatmap": torch.rand(b, 32, 32),
        "passage_idx": torch.tensor([[0, 1, 2], [2, 3, 4]]), "presence": torch.tensor([1.0, 0.0]),
        "log_diameter": torch.tensor([2.0, 0.0]), "nodule_type": torch.tensor([1, 0]), "margin": torch.tensor([3, 0]),
        "lung_rads": torch.tensor([5, 0]), "malignancy": torch.tensor([1.0, 0.0]), "lymph": torch.tensor([0.0, 0.0]),
        "histology": torch.tensor([4, 0]), "histology_mask": torch.tensor([1.0, 0.0]),
    }


def test_forward_backward_produces_finite_loss():
    cfg = tiny_config()
    model = MedFusionVLM(cfg, 60, 0, torch.randint(4, 60, (6, 16)))
    batch = fake_batch()
    out = model(batch, 0.1, 0.1)
    loss, terms = multitask_loss(out, batch, cfg["training"]["loss_weights"], 0, 0.05)
    loss.backward()
    assert torch.isfinite(loss)
    assert out["heatmap"].shape == (2, 32, 32)
    assert set(terms) >= {"lm", "heatmap", "lung_rads", "histology", "alignment"}


def test_cached_generation_matches_full_forward():
    torch.manual_seed(1)
    decoder = RetrievalAugmentedDecoder(40, 32, 4, 2, 32, 0.0, 0).eval()
    memory, pad = torch.randn(2, 9, 32), torch.zeros(2, 9, dtype=torch.bool)
    prompt = torch.tensor([[2, 5, 6], [2, 5, 6]])
    ids, _ = decoder.generate(prompt, memory, pad, 8, eos_id=3)
    logits, _ = decoder(torch.cat([prompt, ids], 1)[:, :~0], memory, pad)
    assert torch.equal(logits[:, prompt.shape[1] + ~0:].argmax(~0), ids)


def test_repository_contains_no_hyphens_or_dashes():
    report = audit_tree(PROJECT_ROOT)
    assert report["violation_count"] == 0, report["violations"][:5]
