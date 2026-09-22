"""A completed-job count alone must not hide a missing configuration."""

import pytest
from summarize import check_job_coverage


def plan_and_rows():
    plan = dict(
        cases=[dict(name="h1")],
        workers=[1],
        repeats=1,
        variants=["legacy-default", "pr-finite"],
        rounds=4,
    )
    rows = [
        dict(case="h1", workers=1, repeat=0, variant=v, status="complete", rounds=4)
        for v in plan["variants"]
    ]
    return plan, rows


def test_complete_matrix_accepted():
    plan, rows = plan_and_rows()
    check_job_coverage(plan, rows)


def test_duplicate_cannot_replace_missing_candidate():
    plan, rows = plan_and_rows()
    with pytest.raises(ValueError, match="duplicated"):
        check_job_coverage(plan, [rows[0], rows[0]])


def test_short_batch_rejected():
    plan, rows = plan_and_rows()
    rows[1]["rounds"] = 3
    with pytest.raises(ValueError, match="round count"):
        check_job_coverage(plan, rows)
