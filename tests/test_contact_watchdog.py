"""Only infrastructure failures may trigger unattended retries."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("contact_watchdog", Path(__file__).parents[1] / "scripts/watch_contact_training.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


@pytest.mark.parametrize("message", ["worker_failed", "Node lost", "EndpointConnectionError", "ReadTimeoutError",
                                     "Watchdog caught collective operation timeout"])
def test_transient_infrastructure_errors_can_resume(message):
    assert watchdog.retryable_failure(message)


@pytest.mark.parametrize("message", ["worker_failed: OOMKilled", "nonfinite gradient; connection reset by peer",
                                     "DataLoader worker lost", "CUDA illegal memory access", "SIGABRT",
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
