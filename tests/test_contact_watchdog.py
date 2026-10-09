"""Only infrastructure failures may trigger unattended retries."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("contact_watchdog", Path(__file__).parents[1] / "scripts/watch_contact_training.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


@pytest.mark.parametrize("message", ["worker_failed", "Node lost", "EndpointConnectionError", "ReadTimeoutError",
                                     "Watchdog caught collective operation timeout",
                                     "Watchdog caught collective operation timeout\n  File torch/utils/data/dataloader.py"])
def test_transient_infrastructure_errors_can_resume(message):
    assert watchdog.retryable_failure(message)


@pytest.mark.parametrize("message", ["worker_failed: OOMKilled", "nonfinite gradient; connection reset by peer",
                                     "DataLoader worker lost", "DataLoader timed out after 300 seconds; worker_failed",
                                     "CUDA illegal memory access", "SIGABRT",
                                     "FileNotFoundError: temporarily unavailable", "AssertionError"])
def test_scientific_data_and_unclear_failures_need_diagnosis(message):
    assert not watchdog.retryable_failure(message)


def test_rescheduling_or_restaging_is_not_a_stalled_old_trainer():
    old = [{"timestamp": 100}]
    assert not watchdog.stalled_training({"tasks": ["building"], "ranks": old}, 2000)
    assert not watchdog.stalled_training({"tasks": ["running"], "ranks": old,
                                         "attempt_started_at": 1500}, 3000)
    assert watchdog.stalled_training({"tasks": ["running"], "ranks": old,
                                     "attempt_started_at": 50}, 2000)
    assert not watchdog.stalled_training({"tasks": ["running"], "ranks": [{"timestamp": 1900}]}, 2000)


def test_time_budget_stop_requires_durable_finished_artifacts():
    snapshot = {"result": {"step": 19000, "finished_steps": False, "wall_seconds": 345601},
                "checkpoint": {"step": 19000}, "checkpoint_bytes": 5795897603,
                "wandb": {"state": "finished"}, "training_budget_seconds": 345600}
    assert watchdog.completion_status(snapshot) == "budget_exhausted"
    assert watchdog.completion_status({**snapshot, "checkpoint_bytes": 0}) == "needs_attention"
    assert watchdog.completion_status({**snapshot, "wandb": {"state": "running"}}) == "needs_attention"
    snapshot["result"]["finished_steps"] = True
    assert watchdog.completion_status(snapshot) == "completed"
