import inspect
from collections.abc import Sequence

import brainpy.math as bm
import numpy as np
from tqdm import tqdm

from ..models.basic.cann import BaseCANN, BaseCANN1D, BaseCANN2D
from ..typing import Iext_pair_type, Iext_type, time_type
from ._base import Task

__all__ = [
    "PopulationCoding1D",
    "TemplateMatching1D",
    "SmoothTracking1D",
    "PopulationCoding2D",
    "TemplateMatching2D",
    "SmoothTracking2D",
]


class TrackingTask(Task):
    """
    A task for simulating the tracking of external inputs in an n-D CANN.

    This class generates a complete time-series of external input stimuli based on
    a predefined sequence of input positions and their corresponding durations.
    It is designed to provide a consistent and repeatable input protocol for
    testing and analyzing Continuous Attractor Neural Network (CANN) models.

    The primary output is the `Iext_sequence` attribute, which is a NumPy array
    representing the input vector at each time step of the simulation.
    """

    def __init__(
        self,
        ndim: int,
        config: dict = None,
        **kwargs,
    ):
        """Initializes the tracking task and pre-computes the input sequence.

        This constructor sets up the simulation parameters and, most importantly,
        calls the internal `_make_Iext_sequence` method to generate the full
        stimulus protocol that will be used during the task run.

        Args:
            ndim (int): The dimensionality of the continuous attractor network space.
            config (dict, optional): A dictionary containing the parameters for the
                tracking simulation. Expected keys are:
                - Iext (Sequence[float]): A sequence of positions for the
                  external input stimulus.
                - duration (Sequence[float]): A sequence of durations, where each
                  duration corresponds to an input position in `Iext`.
                - time_step (float, optional): The simulation time step.
                  Defaults to 0.1.
                - cann_instance (BaseCANN): An instance of the CANN model
                    to be used for generating the stimulus patterns.
        """
        super().__init__()
        assert config is not None
        self.duration = config.get("duration", [])
        self.Iext = config.get("Iext", [])
        self.ndim = ndim

        # Simulation time control
        self.current_step = 0
        self.time_step = config.get("time_step", 0.1)
        self.total_duration = np.sum(self.duration)
        self.total_steps = np.ceil(self.total_duration / self.time_step).astype(dtype=int)

        self.run_steps = bm.arange(0, self.total_duration, self.time_step)

        # checks
        if self.Iext is None or not isinstance(self.Iext, Sequence):
            raise ValueError("Configuration must include 'Iext' as a sequence of input positions.")
        if self.duration is None or not isinstance(self.duration, Sequence):
            raise ValueError("Configuration must include 'duration' as a sequence of time values.")

        # cann_instance (now supports Place Cell and Grid Cell models too)
        cann_instance = config.get("cann_instance", None)
        if cann_instance is None:
            raise ValueError("Configuration must include 'cann_instance' as a model instance.")

        # Check if instance has required interface methods
        if not hasattr(cann_instance, "get_stimulus_by_pos"):
            raise ValueError(
                "Model instance must have 'get_stimulus_by_pos' method. "
                "This includes CANN models, Place Cell models, and Grid Cell models."
            )

        # For backward compatibility, still check BaseCANN but allow others
        if not (isinstance(cann_instance, BaseCANN) or hasattr(cann_instance, "shape")):
            raise ValueError(
                "Model instance must be a BaseCANN or have compatible interface "
                "(shape attribute and get_stimulus_by_pos method)."
            )
        self.shape = cann_instance.shape
        self.get_stimulus_by_pos = cann_instance.get_stimulus_by_pos

        # Analyze model interface for theta modulation support
        self._analyze_model_interface(cann_instance)

    def _analyze_model_interface(self, cann_instance):
        """
        Analyze the model's get_stimulus_by_pos method to determine if it requires time input.
        This enables automatic support for theta-modulated models without breaking compatibility.

        Args:
            cann_instance: The model instance to analyze.
        """
        try:
            # Inspect the method signature to check parameter count
            sig = inspect.signature(cann_instance.get_stimulus_by_pos)
            # Regular models have 1 parameter (pos), theta models have 2 (pos, time)
            param_count = len(sig.parameters)
            has_theta_attributes = hasattr(cann_instance, "theta_freq") and hasattr(
                cann_instance, "theta_amp"
            )

            # Model needs time input if it has more than 1 parameter AND theta attributes
            self.needs_time_input = param_count > 1 and has_theta_attributes
        except Exception:
            # Fallback: check for theta-related attributes only
            # Theta-modulated models typically have theta_freq attribute
            self.needs_time_input = hasattr(cann_instance, "theta_freq")

    def _make_Iext_sequence(self):
        """
        Creates a time-series array of external input positions.
        This method generates a step-function sequence where each input `Iext[i]` is held constant
        for the corresponding `duration[i]`.

        Returns:
            Quantity or Array: An array representing the external input position at each time step.
        """
        Iext_sequence = np.zeros((self.total_steps, self.ndim), dtype=float)

        start_step = 0
        dur_steps = [int(dur / self.time_step) for dur in self.duration]
        for num_steps, iext_val in zip(dur_steps, self.Iext, strict=False):
            end_step = start_step + num_steps
            Iext_sequence[start_step:end_step, :] = iext_val
            start_step = end_step
        # If total duration is not perfectly divisible, fill the remainder with the last value.
        if start_step < self.total_steps:
            Iext_sequence[start_step:] = self.Iext[-1]
        return Iext_sequence

    def get_data(self, progress_bar: bool = True):
        """
        Generates the task data by creating a sequence of external inputs
        based on the provided `Iext` and `duration` parameters.

        Automatically handles both regular models and theta-modulated models
        by intelligently passing time parameters when needed.
        """
        self.Iext_sequence = self._make_Iext_sequence()

        shape = (len(self.Iext_sequence), *self.shape)
        data = np.zeros(shape, dtype=float)

        # Initialize time tracking for theta-modulated models
        current_time = 0.0

        for i, pos in tqdm(
            enumerate(self.Iext_sequence),
            desc=f"<{type(self).__name__}> Generating Task data",
            disable=not progress_bar,
        ):
            if self.needs_time_input:
                # Theta-modulated model: pass both position and time
                data[i] = self.get_stimulus_by_pos(pos, current_time)
                current_time += self.time_step
            else:
                # Regular model: pass only position
                data[i] = self.get_stimulus_by_pos(pos)

        self.data = data

    def show_data(
        self,
        show=True,
        save_path=None,
    ):
        raise NotImplementedError(
            "The show_data method is not implemented for TrackingTask. "
            "Please implement this method in subclasses to visualize the task data."
        )


class PopulationCoding(TrackingTask):
    """
    Population coding task for n-D continuous attractor networks.
    In this task, a stimulus is presented for a specific duration, preceded and followed by
    periods of no stimulation, to test the network's ability to form and maintain a memory bump.
    """

    def __init__(
        self,
        cann_instance: BaseCANN,
        ndim: int,
        before_duration: time_type,
        after_duration: time_type,
        Iext: Iext_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Population Coding task.

        Args:
            cann_instance (BaseCANN): An instance of the 1D CANN model.
            ndim (int): The dimensionality of the continuous attractor network.
            before_duration (float | Quantity): Duration of the initial period with no stimulus.
            after_duration (float | Quantity): Duration of the final period with no stimulus.
            Iext (float | Quantity): The position of the external input during the stimulation period.
            duration (float | Quantity): The duration of the stimulation period.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        # The task is structured as: no input -> input -> no input.
        # The base class handles this by taking sequences. Here, we provide dummy values for the
        # 'no input' periods, as the `update` method will handle turning off the input.
        super().__init__(
            ndim=ndim,
            config={
                "cann_instance": cann_instance,
                "Iext": (Iext, Iext, Iext),  # Repeated for before, during, and after phases.
                "duration": (before_duration, duration, after_duration),  # Duration for each phase.
                "time_step": time_step,  # Time step for the simulation.
            },
        )
        self.before_duration = before_duration
        self.after_duration = after_duration

    def get_data(self, progress_bar: bool = True):
        """
        Generate task data with a constant stimulus during specified time window.

        Creates input sequence where stimulus is only present during the interval
        [before_duration, total_duration - after_duration], with zeros elsewhere.

        Args:
            progress_bar: Whether to display progress information (default: True)

        Returns:
            None. Sets self.data attribute with shape (total_steps, *network_shape)
        """
        self.Iext_sequence = self._make_Iext_sequence()

        shape = (self.total_steps,) + self.shape
        data = np.zeros(shape, dtype=float)

        # Determine the time boundaries for applying the stimulus.
        start_time_step = int(self.before_duration / self.time_step)
        end_time_step = int((self.total_duration - self.after_duration) / self.time_step)
        stimulus = self.get_stimulus_by_pos(self.Iext_sequence[start_time_step])

        # for i in tqdm(
        #     range(start_time_step, end_time_step),
        #     desc=f"<{type(self).__name__}>Generating Task data",
        #     disable=not progress_bar
        # ):
        if progress_bar:
            print(f"<{type(self).__name__}>Generating Task data(No For Loop)")
        data[start_time_step:end_time_step] = stimulus

        self.data = data


class TemplateMatching(TrackingTask):
    """
    Template matching task for n-D continuous attractor networks.

    This task presents a constant stimulus template with Gaussian noise added at each
    time step. The network must denoise the input and converge to the clean template,
    testing its attractor dynamics and noise robustness.

    The noisy stimulus is generated as: stimulus + 0.1 * A * randn()
    where A is the network's stimulus amplitude parameter.
    """

    def __init__(
        self,
        cann_instance: BaseCANN,
        ndim: int,
        Iext: Iext_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Template Matching task.

        Args:
            cann_instance (BaseCANN): An instance of the 1D CANN model.
            ndim (int): The dimensionality of the continuous attractor network.
            Iext (float | Quantity): The position of the external input.
            duration (float | Quantity): The duration for which the noisy stimulus is presented.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        super().__init__(
            ndim=ndim,
            config={
                "cann_instance": cann_instance,
                "Iext": (Iext,),  # Single input position for the template matching task.
                "duration": (duration,),  # Single duration for the stimulus.
                "time_step": time_step,  # Time step for the simulation.
            },
        )
        self.A = cann_instance.A  # The amplitude of the noise to be added.

    def get_data(self, progress_bar: bool = True):
        """
        Generate noisy stimulus data for template matching task.

        Creates a sequence where each time step contains the same template pattern
        with different random Gaussian noise added.

        Args:
            progress_bar: Whether to display progress bar during generation (default: True)

        Returns:
            None. Sets self.data attribute with shape (total_steps, *network_shape)
        """
        self.Iext_sequence = self._make_Iext_sequence()

        shape = (self.total_steps,) + self.shape
        data = np.zeros(shape, dtype=float)

        # Generate the stimulus pattern for the given input position.
        stimulus = self.get_stimulus_by_pos(self.Iext_sequence[0])

        # Add noise to the stimulus for each time step.
        for i in tqdm(
            range(self.total_steps),
            desc=f"<{type(self).__name__}>Generating Task data",
            disable=not progress_bar,
        ):
            noise = 0.1 * self.A * np.random.randn(*self.shape)
            data[i] = stimulus + noise

        self.data = data


class SmoothTracking(TrackingTask):
    """
    Smooth tracking task for n-D continuous attractor networks.
    This task provides an external input that moves smoothly over time, testing the network's
    ability to track a continuously changing stimulus.
    """

    def __init__(
        self,
        cann_instance: BaseCANN,
        ndim: int,
        Iext: Sequence[Iext_type],
        duration: Sequence[time_type],
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Smooth Tracking task.

        Args:
            cann_instance (BaseCANN): An instance of the 1D CANN model.
            Iext (Sequence[float | Quantity]): A sequence of keypoint positions for the input.
            duration (Sequence[float | Quantity]): The duration of each segment of smooth movement.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        assert len(tuple(Iext)) == (len(tuple(duration)) + 1), (
            "Iext must have one more element than duration to define start and end points for each segment."
        )
        super().__init__(
            ndim=ndim,
            config={
                "cann_instance": cann_instance,
                "Iext": Iext,  # Sequence of keypoint positions for the input.
                "duration": duration,  # Sequence of durations for each segment.
                "time_step": time_step,  # Time step for the simulation.
            },
        )

    def _make_Iext_sequence(self):
        """
        Creates a time-series of external input positions that smoothly transitions
        between the keypoints defined in `self.Iext`.
        The output is an array of shape (total_steps, ndim).
        """
        # The output sequence now has a shape of (total_steps, ndim) to hold coordinates.
        Iext_sequence = np.zeros((self.total_steps, self.ndim), dtype=float)
        start_step = 0

        if self.ndim == 1:
            for i, dur in enumerate(self.duration):
                num_steps = int(dur / self.time_step)
                if num_steps == 0:
                    continue
                end_step = start_step + num_steps
                Iext_sequence[start_step:end_step] = np.linspace(
                    self.Iext[i], self.Iext[i + 1], num_steps
                ).reshape(-1, 1)
                start_step = end_step
            if start_step < self.total_steps:
                Iext_sequence[start_step:] = self.Iext[-1]
        else:
            for i, dur in enumerate(self.duration):
                num_steps = int(dur / self.time_step)
                if num_steps == 0:
                    continue
                end_step = start_step + num_steps

                # Define start and end points (which are now tuples/vectors) for interpolation
                start_pos = self.Iext[i]
                end_pos = self.Iext[i + 1]

                # Interpolate each dimension independently
                for d in range(self.ndim):
                    start_d = start_pos[d]
                    end_d = end_pos[d]
                    Iext_sequence[start_step:end_step, d] = np.linspace(start_d, end_d, num_steps)

                start_step = end_step

            # Fill any remaining steps with the final position.
            if start_step < self.total_steps:
                # self.Iext[-1] is a tuple of shape (ndim,), which will be broadcast correctly.
                Iext_sequence[start_step:, :] = self.Iext[-1]

        return Iext_sequence


class PopulationCoding1D(PopulationCoding):
    """
    Population coding task for 1D continuous attractor networks.
    In this task, a stimulus is presented for a specific duration, preceded and followed by
    periods of no stimulation, to test the network's ability to form and maintain a memory bump.
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        before_duration: time_type,
        after_duration: time_type,
        Iext: Iext_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Population Coding task.

        Args:
            cann_instance (BaseCANN1D): An instance of the 1D CANN model.
            before_duration (float | Quantity): Duration of the initial period with no stimulus.
            after_duration (float | Quantity): Duration of the final period with no stimulus.
            Iext (float | Quantity): The position of the external input during the stimulation period.
            duration (float | Quantity): The duration of the stimulation period.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        # The task is structured as: no input -> input -> no input.
        # The base class handles this by taking sequences. Here, we provide dummy values for the
        # 'no input' periods, as the `update` method will handle turning off the input.
        super().__init__(
            cann_instance=cann_instance,
            ndim=1,
            before_duration=before_duration,
            after_duration=after_duration,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )
        self.before_duration = before_duration
        self.after_duration = after_duration


class TemplateMatching1D(TemplateMatching):
    """
    Template matching task for 1D continuous attractor networks.
    This task presents a stimulus with added noise to test the network's ability to
    denoise the input and settle on the correct underlying pattern (template).
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Iext_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Template Matching task.

        Args:
            cann_instance (BaseCANN1D): An instance of the 1D CANN model.
            Iext (float | Quantity): The position of the external input.
            duration (float | Quantity): The duration for which the noisy stimulus is presented.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        super().__init__(
            cann_instance=cann_instance,
            ndim=1,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )


class SmoothTracking1D(SmoothTracking):
    """
    Smooth tracking task for 1D continuous attractor networks.
    This task provides an external input that moves smoothly over time, testing the network's
    ability to track a continuously changing stimulus.
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Sequence[Iext_type],
        duration: Sequence[time_type],
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Smooth Tracking task.

        Args:
            cann_instance (BaseCANN1D): An instance of the 1D CANN model.
            Iext (Sequence[float | Quantity]): A sequence of keypoint positions for the input.
            duration (Sequence[float | Quantity]): The duration of each segment of smooth movement.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        super().__init__(
            cann_instance=cann_instance,
            ndim=1,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )


class CustomTracking1D(TrackingTask):
    """
    A template class for creating custom 1D tracking tasks.
    Users should inherit from this class and implement their own logic for
    `_make_Iext_sequence` and/or `update` to define a new task.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the custom task using the base class constructor."""
        super().__init__(*args, ndim=1, **kwargs)

    def _make_Iext_sequence(self):
        """
        Placeholder for custom input sequence generation.
        This method should be overridden to create a specific time-series of inputs.
        """
        # Example: raise an error to enforce implementation by subclasses.
        raise NotImplementedError("Please implement _make_Iext_sequence for your custom task.")

    def update(self):
        """
        Placeholder for custom update logic.
        This method can be overridden to introduce custom behavior at each time step,
        such as adding specific types of noise or conditional stimuli.
        """
        # Example: raise an error to enforce implementation by subclasses.
        raise NotImplementedError("Please implement the update logic for your custom task.")


class PopulationCoding2D(PopulationCoding):
    """
    Population coding task for 2D continuous attractor networks.
    In this task, a stimulus is presented for a specific duration, preceded and followed by
    periods of no stimulation, to test the network's ability to form and maintain a memory bump.
    """

    def __init__(
        self,
        cann_instance: BaseCANN2D,
        before_duration: time_type,
        after_duration: time_type,
        Iext: Iext_pair_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Population Coding task.

        Args:
            cann_instance (BaseCANN2D): An instance of the 2D CANN model.
            before_duration (float | Quantity): Duration of the initial period with no stimulus.
            after_duration (float | Quantity): Duration of the final period with no stimulus.
            Iext (float | Quantity): The position of the external input during the stimulation period.
            duration (float | Quantity): The duration of the stimulation period.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        # The task is structured as: no input -> input -> no input.
        # The base class handles this by taking sequences. Here, we provide dummy values for the
        # 'no input' periods, as the `update` method will handle turning off the input.
        assert len(Iext) == 2, "Iext must be a tuple of two values for 2D tracking."
        super().__init__(
            cann_instance=cann_instance,
            ndim=2,
            before_duration=before_duration,
            after_duration=after_duration,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )
        self.before_duration = before_duration
        self.after_duration = after_duration


class TemplateMatching2D(TemplateMatching):
    """
    Template matching task for 2D continuous attractor networks.
    This task presents a stimulus with added noise to test the network's ability to
    denoise the input and settle on the correct underlying pattern (template).
    """

    def __init__(
        self,
        cann_instance: BaseCANN2D,
        Iext: Iext_pair_type,
        duration: time_type,
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Template Matching task.

        Args:
            cann_instance (BaseCANN2D): An instance of the 2D CANN model.
            Iext (tuple[float, float] | Quantity): The 2D position of the external input.
            duration (float | Quantity): The duration for which the noisy stimulus is presented.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        assert len(Iext) == 2, "Iext must be a tuple of two values for 2D tracking."
        super().__init__(
            cann_instance=cann_instance,
            ndim=2,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )


class SmoothTracking2D(SmoothTracking):
    """
    Smooth tracking task for 2D continuous attractor networks.
    This task provides an external input that moves smoothly over time, testing the network's
    ability to track a continuously changing stimulus.
    """

    def __init__(
        self,
        cann_instance: BaseCANN2D,
        Iext: Sequence[Iext_pair_type],
        duration: Sequence[time_type],
        time_step: time_type = 0.1,
    ):
        """
        Initializes the Smooth Tracking task.

        Args:
            cann_instance (BaseCANN2D): An instance of the 2D CANN model.
            Iext (Sequence[tuple[float, float] | Quantity]): A sequence of 2D keypoint positions for the input.
            duration (Sequence[float | Quantity]): The duration of each segment of smooth movement.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        super().__init__(
            cann_instance=cann_instance,
            ndim=2,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )


class CustomTracking2D(TrackingTask):
    """
    A template class for creating custom 2D tracking tasks.
    Users should inherit from this class and implement their own logic for
    `_make_Iext_sequence` and/or `update` to define a new task.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the custom task using the base class constructor."""
        super().__init__(*args, ndim=2, **kwargs)

    def _make_Iext_sequence(self):
        """
        Placeholder for custom input sequence generation.
        This method should be overridden to create a specific time-series of inputs.
        """
        # Example: raise an error to enforce implementation by subclasses.
        raise NotImplementedError("Please implement _make_Iext_sequence for your custom task.")

    def update(self):
        """
        Placeholder for custom update logic.
        This method can be overridden to introduce custom behavior at each time step,
        such as adding specific types of noise or conditional stimuli.
        """
        # Example: raise an error to enforce implementation by subclasses.
        raise NotImplementedError("Please implement the update logic for your custom task.")
