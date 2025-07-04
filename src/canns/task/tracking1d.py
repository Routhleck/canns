from typing import Sequence

import brainstate
import brainstate as bst
import brainunit as u
from brainunit import Quantity
from saiunit.math import dtype

from ._base import BaseTask
from ..models.basic.cann1d import BaseCANN1D

__all__ = [
    'PopulationCoding1D',
    'TemplateMatching1D',
    'SmoothTracking1D',
    'CustomTracking1D',
]


class Tracking1D(BaseTask):
    """
    Base class for 1D tracking tasks.
    This class manages a sequence of external inputs applied to a 1D CANN model over time.
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Sequence[float | Quantity],
        duration: Sequence[float | Quantity],
        time_step: float | Quantity = 0.1,
        **kwargs
    ):
        """
        Initializes the 1D tracking task.

        Args:
            cann_instance (BaseCANN1D): An instance of the 1D Continuous Attractor Neural Network model.
            Iext (Sequence[float | Quantity]): A sequence of external input positions.
            duration (Sequence[float | Quantity]): A sequence of durations for each corresponding external input.
            time_step (float | Quantity, optional): The time step for the simulation. Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.cann_instance = cann_instance
        self.duration = duration
        self.Iext = Iext

        # Simulation time control
        self.current_step = 0
        self.time_step = time_step
        self.total_duration = sum(duration)
        self.total_steps = u.math.ceil(self.total_duration / self.time_step).astype(dtype=int)

        # Pre-computes the entire sequence of external inputs for the simulation.
        self.Iext_sequence = self._make_Iext_sequence()

        self.run_steps = u.math.arange(0, self.total_duration, bst.environ.get_dt())

    def init_state(self, *args, **kwargs):
        """
        Initializes the state variables for the simulation run.
        This method should be called at the beginning of a simulation.
        """
        self.current_step = brainstate.State(0, dtype=int)
        self.inputs = brainstate.State(u.math.zeros(self.cann_instance.varshape))

    def _make_Iext_sequence(self):
        """
        Creates a time-series array of external input positions.
        This method generates a step-function sequence where each input `Iext[i]` is held constant
        for the corresponding `duration[i]`.

        Returns:
            Quantity or Array: An array representing the external input position at each time step.
        """
        Iext_sequence = Quantity(u.math.zeros(self.total_steps, dtype=float))
        start_step = 0
        for dur, iext_val in zip(self.duration, self.Iext):
            num_steps = int(dur / self.time_step)
            end_step = start_step + num_steps
            Iext_sequence[start_step:end_step] = iext_val
            start_step = end_step
        # If total duration is not perfectly divisible, fill the remainder with the last value.
        if start_step < self.total_steps:
            Iext_sequence[start_step:] = self.Iext[-1]
        return u.math.maybe_decimal(Iext_sequence)

    def _step(self):
        """
        Performs a single step of the tracking task.
        It retrieves the external input position for the current time step.

        Returns:
            float | bool: The input position for the current step, or False if the simulation is over.
        """
        # Check if the simulation has completed all steps.
        return u.math.where(
            self.current_step.value >= self.total_steps,
            self.Iext_sequence[-1],  # Return the last input position if the end is reached.
            self.Iext_sequence[self.current_step.value]  # Return the pre-calculated input.
        )

    def update(self):
        """
        Updates the task state for the next time step.
        This method gets the next input position and generates the corresponding stimulus pattern
        for the CANN model. It's the main function called by the runner at each step.

        Returns:
            bool: False if the simulation is over, otherwise implicitly returns None.
        """
        pos = self._step()
        if pos is False:
            return False  # Signal to stop the simulation.

        self.current_step.value += 1
        # Convert the position `pos` into a stimulus pattern for the neural network.
        self.inputs.value = self.cann_instance.get_stimulus_by_pos(pos)


class PopulationCoding1D(Tracking1D):
    """
    Population coding task for 1D continuous attractor networks.
    In this task, a stimulus is presented for a specific duration, preceded and followed by
    periods of no stimulation, to test the network's ability to form and maintain a memory bump.
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        before_duration: float | Quantity,
        after_duration: float | Quantity,
        Iext: float | Quantity,
        duration: float | Quantity,
        time_step: float | Quantity = 0.1,
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
            Iext=(Iext, Iext, Iext),  # Position is constant, but only applied during the middle phase.
            duration=(before_duration, duration, after_duration),
            time_step=time_step,
        )
        self.before_duration = before_duration
        self.after_duration = after_duration

    def update(self):
        """
        Updates the task state for the next time step.
        It provides a stimulus only during the central time window and provides zero input otherwise.
        """
        pos = self._step()
        if pos is False:
            return False

        # Determine the time boundaries for applying the stimulus.
        start_stim_step = self.before_duration / self.time_step
        end_stim_step = (self.total_duration - self.after_duration) / self.time_step

        # Apply stimulus only if the current step is within the stimulation window.
        self.inputs.value = u.math.where(
            u.math.logical_and(
                self.current_step.value > start_stim_step,
                self.current_step.value < end_stim_step
            ),
            self.cann_instance.get_stimulus_by_pos(pos),  # Apply stimulus
            u.math.zeros(self.cann_instance.varshape, dtype=float)  # Apply no stimulus
        )
        self.current_step.value += 1


class TemplateMatching1D(Tracking1D):
    """
    Template matching task for 1D continuous attractor networks.
    This task presents a stimulus with added noise to test the network's ability to
    denoise the input and settle on the correct underlying pattern (template).
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: float | Quantity,
        duration: float | Quantity,
        time_step: float | Quantity = 0.1,
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
            Iext=(Iext,),
            duration=(duration,),
            time_step=time_step,
        )

    def update(self):
        """
        Updates the task state for the next time step.
        It generates the stimulus and adds Gaussian noise to it.
        """
        pos = self._step()
        if pos is False:
            return False

        self.current_step.value += 1
        # Generate the base stimulus pattern.
        stimulus = self.cann_instance.get_stimulus_by_pos(pos)
        # Add random noise to the stimulus.
        noise = 0.1 * self.cann_instance.A * bst.random.randn(*self.cann_instance.varshape)
        self.inputs.value = stimulus + noise


class SmoothTracking1D(Tracking1D):
    """
    Smooth tracking task for 1D continuous attractor networks.
    This task provides an external input that moves smoothly over time, testing the network's
    ability to track a continuously changing stimulus.
    """

    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Sequence[float | Quantity],
        duration: Sequence[float | Quantity],
        time_step: float | Quantity = 0.1,
    ):
        """
        Initializes the Smooth Tracking task.

        Args:
            cann_instance (BaseCANN1D): An instance of the 1D CANN model.
            Iext (Sequence[float | Quantity]): A sequence of keypoint positions for the input.
            duration (Sequence[float | Quantity]): The duration of each segment of smooth movement.
            time_step (float | Quantity, optional): The simulation time step. Defaults to 0.1.
        """
        assert len(tuple(Iext)) == (len(tuple(duration)) + 1), \
            "Iext must have one more element than duration to define start and end points for each segment."
        super().__init__(
            cann_instance=cann_instance,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )

    def _make_Iext_sequence(self):
        """
        Creates a time-series of external input positions that smoothly transitions
        between the keypoints defined in `self.Iext`.
        """
        Iext_sequence = Quantity(u.math.zeros(self.total_steps, dtype=float))
        start_step = 0
        for i, dur in enumerate(self.duration):
            num_steps = int(dur / self.time_step)
            end_step = start_step + num_steps

            # Define start and end points for interpolation
            start_pos = self.Iext[i]
            # If there is a next point, interpolate towards it. Otherwise, hold the last position.
            end_pos = self.Iext[i + 1] if i + 1 < len(self.Iext) else self.Iext[-1]

            # Linearly interpolate between the start and end positions for the current segment.
            Iext_sequence[start_step:end_step] = u.math.linspace(start_pos, end_pos, num_steps)
            start_step = end_step

        # Fill any remaining steps with the final position.
        if start_step < self.total_steps:
            Iext_sequence[start_step:] = self.Iext[-1]

        return u.math.maybe_decimal(Iext_sequence)

    def update(self):
        """
        Updates the task state for the next time step.
        This uses the standard update logic from the base class, but operates on the
        smoothly varying input sequence generated by the overridden `_make_Iext_sequence`.
        """
        pos = self._step()
        if pos is False:
            return False

        self.current_step.value += 1
        self.inputs.value = self.cann_instance.get_stimulus_by_pos(pos)


class CustomTracking1D(Tracking1D):
    """
    A template class for creating custom 1D tracking tasks.
    Users should inherit from this class and implement their own logic for
    `_make_Iext_sequence` and/or `update` to define a new task.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the custom task using the base class constructor."""
        super().__init__(*args, **kwargs)

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
