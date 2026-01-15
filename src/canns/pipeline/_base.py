from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Pipeline(ABC):
    """Abstract base class for CANNs pipelines.

    Pipelines orchestrate multi-step workflows (data preparation, model execution,
    visualization, etc.). This base class standardizes how we manage results and
    output directories so derived pipelines can focus on domain-specific logic.
    """

    def __init__(self) -> None:
        self.results: dict[str, Any] | None = None
        self.output_dir: Path | None = None

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Execute the pipeline and return a mapping of generated artifacts."""

    def reset(self) -> None:
        """Reset stored state so the pipeline can be executed again cleanly."""
        self.results = None
        self.output_dir = None

    def prepare_output_dir(self, output_dir: str | Path, *, create: bool = True) -> Path:
        """Validate and optionally create the output directory for derived pipelines."""
        path = Path(output_dir)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        self.output_dir = path
        return path

    def set_results(self, results: dict[str, Any]) -> dict[str, Any]:
        """Store pipeline results and return them for convenient chaining."""
        self.results = results
        return results

    def get_results(self) -> dict[str, Any]:
        """Return stored results or raise if the pipeline has not been executed."""
        if self.results is None:
            raise RuntimeError("Pipeline results are not available; call run() first.")
        return self.results

    def has_results(self) -> bool:
        """Check whether the pipeline has already produced results."""
        return self.results is not None
