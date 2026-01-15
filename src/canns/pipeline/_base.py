from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Pipeline(ABC):
    """Abstract base class for CANNs pipelines.

    Pipelines orchestrate multi-step workflows (data preparation, model execution,
    visualization, etc.). This base class standardizes how we manage results and
    output directories so derived pipelines can focus on domain-specific logic.

    All pipeline subclasses must implement the ``run()`` method to define their
    specific workflow. The base class provides utilities for managing output
    directories, storing results, and resetting state between runs.

    Attributes
    ----------
    results : dict[str, Any] or None
        Dictionary containing pipeline outputs after execution. None if not yet run.
    output_dir : Path or None
        Directory path where pipeline outputs are saved. None if not yet configured.

    Methods
    -------
    run(*args, **kwargs)
        Execute the pipeline workflow (must be implemented by subclasses)
    reset()
        Clear stored results and output directory to prepare for a new run
    prepare_output_dir(output_dir, create=True)
        Validate and optionally create the output directory
    set_results(results)
        Store pipeline results after execution
    get_results()
        Retrieve stored results (raises if pipeline hasn't been run)
    has_results()
        Check whether results are available

    Example
    -------
    Creating a custom pipeline:

    >>> from pathlib import Path
    >>> from canns.pipeline import Pipeline
    >>> 
    >>> class MyAnalysisPipeline(Pipeline):
    ...     def __init__(self, data):
    ...         super().__init__()
    ...         self.data = data
    ...     
    ...     def run(self, output_dir="output", verbose=False):
    ...         # Prepare output directory
    ...         out_path = self.prepare_output_dir(output_dir)
    ...         
    ...         if verbose:
    ...             print(f"Processing data in {out_path}")
    ...         
    ...         # Perform analysis
    ...         result = {"mean": float(self.data.mean()),
    ...                   "std": float(self.data.std())}
    ...         
    ...         # Store and return results
    ...         return self.set_results(result)
    >>> 
    >>> # Use the pipeline
    >>> import numpy as np
    >>> pipeline = MyAnalysisPipeline(np.random.randn(100))
    >>> results = pipeline.run(verbose=True)
    >>> print(results)
    {'mean': 0.05, 'std': 1.02}
    >>> 
    >>> # Check if results are available
    >>> assert pipeline.has_results()
    >>> 
    >>> # Reset for another run
    >>> pipeline.reset()
    >>> assert not pipeline.has_results()

    Notes
    -----
    - Subclasses should call ``self.set_results()`` at the end of ``run()``
    - Use ``prepare_output_dir()`` to set up file output locations
    - Call ``reset()`` between runs if reusing the same pipeline instance
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
