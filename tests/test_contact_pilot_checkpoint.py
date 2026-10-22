"""Check that a long pilot preserves optimizer history and evaluation provenance."""
from dataclasses import dataclass

import torch

from helico.contact_pilot import collect_evaluations, save_checkpoint, write_csv


@dataclass
class Config:
    predict_contacts: bool = True


class SmallModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1)
        self.config = Config()


def update(model, optimizer):
    optimizer.zero_grad()
    model(torch.tensor([[1., 2.]])).square().sum().backward()
    optimizer.step()


def test_checkpoint_preserves_adam_history_for_next_update(tmp_path):
    model = SmallModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    update(model, optimizer)
    path = tmp_path / "step-1.pt"
    save_checkpoint(path, model, optimizer, 1, "masked", 8)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["step"] == 1
    assert checkpoint["world_size"] == 8
    assert checkpoint["model_config"] == {"predict_contacts": True}
    assert not path.with_suffix(".tmp").exists()
    restored = SmallModel()
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.5)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    update(model, optimizer)
    update(restored, restored_optimizer)
    for original, resumed in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, resumed, rtol=0, atol=0)


def test_collection_keeps_requested_phases_and_rank_records(tmp_path):
    import csv
    for phase, step in (("initial", 0), ("step-256", 256), ("final", 2048)):
        for rank in range(2):
            write_csv(tmp_path / f"{phase}-rank{rank}.csv",
                      [{"phase": phase, "step": step, "pdb_id": f"target-{rank}"}])
    collect_evaluations(tmp_path, ["initial", "step-256"])
    with (tmp_path / "evaluation.csv").open() as f:
        records = list(csv.DictReader(f))
    assert len(records) == 4
    assert {r["phase"] for r in records} == {"initial", "step-256"}
    assert {r["pdb_id"] for r in records} == {"target-0", "target-1"}
