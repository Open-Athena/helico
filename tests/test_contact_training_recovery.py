import json
import os
import subprocess
import sys

import fsspec
import pytest

from helico.coreweave_worker import run_trainer
from helico.train_contacts import checkpoint_due, check_resume_config, weighted_draws


def test_recovery_keeps_scientific_configuration_fixed():
    before = dict(run_name="old", output_uri="old", steps=20000, seed=2201,
                  crop_size=384, lr=2e-5, gpus=8, accumulation=4)
    after = {**before, "run_name": "new", "output_uri": "new", "resume_uri": "old",
             "startup_save_every": 50, "loader_timeout_seconds": 300}
    check_resume_config(before, after)
    check_resume_config(before, {**after, "gpus": 4, "accumulation": 8})
    for key, value in [("lr", 1e-4), ("crop_size", 256), ("gpus", 4), ("seed", 1)]:
        with pytest.raises(ValueError):
            check_resume_config(before, {**after, key: value})


def test_startup_saves_bound_lost_work_without_retaining_400_checkpoints():
    config = dict(save_every=250, startup_save_every=50, warmup_steps=500)
    saved = [s for s in range(1, 20001) if checkpoint_due(s, config)]
    assert saved[:11] == [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    assert saved[11] == 750 and saved[-1] == 20000
    assert len(saved) == 89


def test_repartitioning_preserves_every_remaining_global_draw():
    def stream(world, accumulation):
        return sorted((draw, index) for rank in range(world)
                      for index, draw in weighted_draws([1., 3., 2.], 9, accumulation,
                                                        world, rank, 2, 2201))
    assert stream(8, 4) == stream(4, 8)


def test_native_child_failure_preserves_stdout_stderr_and_exit_record(tmp_path):
    destination = tmp_path / "durable"
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_trainer([sys.executable, "-c",
                     "import sys; print('before abort'); print('failure detail',file=sys.stderr); sys.exit(7)"],
                    os.environ.copy(), tmp_path, {"output_uri": str(destination)},
                    diagnostics_fs=fsspec.filesystem("file", auto_mkdir=True))
    assert error.value.returncode == 7
    records = list(destination.glob("diagnostics/*/exit.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["returncode"] == 7
    log = records[0].with_name("trainer.log").read_text()
    assert "before abort" in log and "failure detail" in log
