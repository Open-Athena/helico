"""Training must not silently enter the opportunistic or admin-only bands."""
import pytest

from helico.coreweave import job_priority


def test_sustained_training_defaults_to_normal_research_priority():
    assert job_priority({"mode": "train"}) == 2
    assert job_priority({"mode": "mirror"}) == 3
    assert job_priority({"mode": "train", "priority_band": "batch"}) == 3


@pytest.mark.parametrize("band", ["system", "production", 1, 4])
def test_admin_priority_is_not_a_recovery_option(band):
    with pytest.raises(ValueError, match="only interactive or batch"):
        job_priority({"mode": "train", "priority_band": band})
