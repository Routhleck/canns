from pathlib import Path

import pytest

from canns.pipeline import Pipeline


class DummyPipeline(Pipeline):
    def run(self, output_dir: str | Path, *, flag: bool = False) -> dict[str, str | bool]:
        path = self.prepare_output_dir(output_dir)
        result = {"output": str(path), "flag": flag}
        return self.set_results(result)


def test_pipeline_helpers_manage_state(tmp_path):
    pipeline = DummyPipeline()
    output_dir = tmp_path / "results"

    assert pipeline.output_dir is None
    assert not pipeline.has_results()

    results = pipeline.run(output_dir=output_dir, flag=True)

    assert output_dir.exists()
    assert pipeline.output_dir == output_dir
    assert results == {"output": str(output_dir), "flag": True}
    assert pipeline.get_results() == results
    assert pipeline.has_results()

    pipeline.reset()
    assert pipeline.output_dir is None
    assert not pipeline.has_results()

    with pytest.raises(RuntimeError):
        pipeline.get_results()
