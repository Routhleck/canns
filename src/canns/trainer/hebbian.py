from ..models import BrainInspiredModel
from ..models.brain_inspired.hopfield import AmariHopfieldNetwork
from .progress import create_progress_reporter

__all__ = ["HebbianTrainer"]


class HebbianTrainer:
    """
    Enhanced Hebbian trainer for Hopfield networks with progress reporting.

    This trainer implements basic Hebbian learning with advanced progress reporting
    capabilities, batch prediction, and flexible configuration options.
    """

    def __init__(
        self,
        model: BrainInspiredModel,
        progress_mode: str = "auto",
        show_iteration_progress: bool = False,  # Default to False for cleaner display
        compiled_prediction: bool = True,
        **progress_kwargs,
    ):
        """
        Initialize Hebbian trainer.

        Args:
            model: The model to train
            progress_mode: Progress reporting mode ("auto", "tqdm", "silent")
            show_iteration_progress: Whether to show progress for individual pattern convergence
            compiled_prediction: Whether to use compiled prediction by default (faster but no iteration progress)
            **progress_kwargs: Additional arguments for progress reporter
        """
        self.model = model
        self.progress_reporter = create_progress_reporter(progress_mode, **progress_kwargs)
        self.show_iteration_progress = show_iteration_progress
        self.compiled_prediction = compiled_prediction

    def train(self, train_data):
        """Train the model using Hebbian learning."""
        self.model.apply_hebbian_learning(train_data)

    def predict(
        self,
        pattern,
        num_iter: int = 20,
        compiled: bool | None = None,
        show_progress: bool | None = None,
        convergence_threshold: float = 1e-10,
    ):
        """
        Predict a single pattern.

        Args:
            pattern: Input pattern to predict
            num_iter: Maximum number of iterations
            compiled: Override default compiled setting
            show_progress: Override default progress setting
            convergence_threshold: Energy change threshold for convergence

        Returns:
            Predicted pattern
        """
        # Use defaults if not specified
        if compiled is None:
            compiled = self.compiled_prediction
        if show_progress is None:
            show_progress = self.show_iteration_progress

        # Create progress callback if needed
        progress_callback = None
        pbar = None

        if show_progress and not compiled:
            # Configure progress bar for better nesting
            has_active_bars = hasattr(self.progress_reporter, '_active_bars') and len(self.progress_reporter._active_bars) > 0
            pbar_kwargs = {"ncols": 80, "leave": False} if has_active_bars else {}
            pbar = self.progress_reporter.start_task("Converging", total=num_iter, **pbar_kwargs)

            def progress_callback(iteration, energy, converged, energy_change):
                # Update with simpler format to avoid clutter
                status_icon = "✓" if converged else "→"
                energy_str = f"{energy:.0f}" if abs(energy) > 1000 else f"{energy:.3f}"
                
                self.progress_reporter.update(
                    pbar,
                    1,
                    E=energy_str,
                    st=status_icon,
                )
                if converged:
                    # Fill remaining iterations to show completion
                    remaining = num_iter - iteration
                    if remaining > 0:
                        self.progress_reporter.update(pbar, remaining)

        try:
            result = self.model.predict(
                pattern,
                num_iter=num_iter,
                compiled=compiled,
                progress_callback=progress_callback,
                convergence_threshold=convergence_threshold,
            )
        finally:
            if pbar is not None:
                self.progress_reporter.finish_task(pbar)

        return result

    def predict_batch(
        self,
        patterns: list,
        num_iter: int = 20,
        compiled: bool | None = None,
        show_sample_progress: bool = True,
        show_iteration_progress: bool | None = None,
        convergence_threshold: float = 1e-10,
    ) -> list:
        """
        Predict multiple patterns with progress reporting.

        Args:
            patterns: List of input patterns to predict
            num_iter: Maximum number of iterations per pattern
            compiled: Override default compiled setting
            show_sample_progress: Whether to show progress across samples
            show_iteration_progress: Override default iteration progress setting
            convergence_threshold: Energy change threshold for convergence

        Returns:
            List of predicted patterns
        """
        # Use defaults if not specified
        if compiled is None:
            compiled = self.compiled_prediction
        if show_iteration_progress is None:
            show_iteration_progress = self.show_iteration_progress

        results = []

        # Create sample-level progress bar
        sample_pbar = None
        if show_sample_progress:
            sample_pbar = self.progress_reporter.start_task(
                "Processing samples", total=len(patterns)
            )

        try:
            for i, pattern in enumerate(patterns):
                # Predict single pattern
                result = self.predict(
                    pattern,
                    num_iter=num_iter,
                    compiled=compiled,
                    show_progress=show_iteration_progress,
                    convergence_threshold=convergence_threshold,
                )
                results.append(result)

                # Update sample progress
                if sample_pbar is not None:
                    self.progress_reporter.update(sample_pbar, 1, sample=f"{i + 1}/{len(patterns)}")

        finally:
            if sample_pbar is not None:
                self.progress_reporter.finish_task(sample_pbar)

        return results

    def configure_progress(
        self,
        progress_mode: str = None,
        show_iteration_progress: bool = None,
        compiled_prediction: bool = None,
        **progress_kwargs,
    ):
        """
        Reconfigure progress reporting settings.

        Args:
            progress_mode: New progress reporting mode ("auto", "tqdm", "silent")
            show_iteration_progress: Whether to show iteration progress
            compiled_prediction: Whether to use compiled prediction by default
            **progress_kwargs: Additional arguments for progress reporter
        """
        if progress_mode is not None:
            self.progress_reporter = create_progress_reporter(progress_mode, **progress_kwargs)

        if show_iteration_progress is not None:
            self.show_iteration_progress = show_iteration_progress

        if compiled_prediction is not None:
            self.compiled_prediction = compiled_prediction
