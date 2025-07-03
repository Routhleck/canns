from typing import Sequence

import brainstate
import jax
import brainstate as bst
import brainunit as u
from brainunit import Quantity

from ._base import BaseTask
from ..models.basic.cann1d import BaseCANN1D

__all__ = [
    'PopulationCoding1D',
    'TemplateMatching1D',
    'SmoothTracking1D',
    'CustomTracking1D',
]

class Tracking1D(BaseTask):
    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Sequence[float | Quantity],
        duration: Sequence[float | Quantity],
        time_step: float | Quantity = 0.1,
        **kwargs
    ):
        """
        Base class for 1D tracking tasks.
        """
        super().__init__(**kwargs)
        self.cann_instance = cann_instance
        self.duration = duration
        self.Iext = Iext

        self.current_step = 0
        self.time_step = time_step
        self.total_duration = sum(duration)
        self.total_steps = int(self.total_duration / self.time_step)

        self.Iext_sequence = self._make_Iext_sequence()

        self.run_steps = u.math.arange(0, self.total_duration, bst.environ.get_dt())

    def init_state(self, *args, **kwargs):
        self.current_step = brainstate.State(0)
        self.inputs = brainstate.State(u.math.zeros(self.cann_instance.varshape))

    def _make_Iext_sequence(self):
        """
        Create a sequence of external inputs based on the provided Iext and duration.
        """
        Iext_sequence = Quantity(u.math.zeros(self.total_steps, dtype=float))
        start = 0
        end = 0
        for duration, Iext in zip(self.duration, self.Iext):
            end = int(duration / self.time_step)
            Iext_sequence[start:end] = Iext
            start = end
        if end < self.total_steps:
            Iext_sequence[end:] = self.Iext[-1]
        return u.math.maybe_decimal(Iext_sequence)


    def _step(self):
        """
        Perform a single step of the tracking task.
        """
        return u.math.where(
            self.current_step.value >= self.total_steps,
            False,
            self.Iext_sequence[self.current_step.value]
        )

    def update(self):
        """
        Get the next input for the CANN instance.
        """
        pos = self._step()
        if pos is False:
            return False

        self.current_step.value += 1
        self.inputs.value = self.cann_instance.get_stimulus_by_pos(pos)



class PopulationCoding1D(Tracking1D):
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
        Population coding task for 1D continuous attractor networks.
        """
        super().__init__(
            cann_instance=cann_instance,
            Iext=(Iext, Iext, Iext),
            duration=(before_duration, duration, after_duration),
            time_step=time_step,
        )
        self.before_duration = before_duration
        self.after_duration = after_duration

    def update(self):
        """
        Get the next input for the CANN instance.
        """
        pos = self._step()
        if pos is False:
            return False

        self.inputs.value = u.math.where(
            u.math.logical_and(
                self.current_step.value > self.before_duration / self.time_step,
                self.current_step.value < (self.total_duration - self.after_duration) / self.time_step
            ),
            self.cann_instance.get_stimulus_by_pos(pos),
            u.math.zeros(self.cann_instance.varshape, dtype=float)
        )
        self.current_step.value += 1


class TemplateMatching1D(Tracking1D):
    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: float | Quantity,
        duration: float | Quantity,
        time_step: float | Quantity = 0.1,
    ):
        """
        Template matching task for 1D continuous attractor networks.
        """
        super().__init__(
            cann_instance=cann_instance,
            Iext=(Iext,),
            duration=(duration,),
            time_step=time_step,
        )

    def update(self):
        """
        Get the next input for the CANN instance.
        """
        pos = self._step()
        if pos is False:
            return False

        self.current_step.value += 1
        self.inputs.value = self.cann_instance.get_stimulus_by_pos(pos)
        self.inputs.value += 0.1 * self.cann_instance.A * bst.random.randn(*self.cann_instance.varshape)

class SmoothTracking1D(Tracking1D):
    def __init__(
        self,
        cann_instance: BaseCANN1D,
        Iext: Sequence[float | Quantity],
        duration: Sequence[float | Quantity],
        time_step: float | Quantity = 0.1,
    ):
        """
        Smooth tracking task for 1D continuous attractor networks.
        """
        super().__init__(
            cann_instance=cann_instance,
            Iext=Iext,
            duration=duration,
            time_step=time_step,
        )

    def _make_Iext_sequence(self):
        """
        Create a sequence of external inputs that smoothly transitions from Iext_start to Iext_end.
        """
        Iext_sequence = Quantity(u.math.zeros(self.total_steps, dtype=float))
        start = 0
        end = 0
        for i, duration in enumerate(self.duration):
            end = int(duration / self.time_step)
            Iext_sequence[start:end] = u.math.linspace(
                self.Iext[i], self.Iext[i + 1] if i + 1 < len(self.Iext) else self.Iext[-1], end - start
            )
            start = end
        if end < self.total_steps:
            Iext_sequence[end:] = self.Iext[-1]
        return u.math.maybe_decimal(Iext_sequence)

        # return u.math.linspace(self.Iext_start, self.Iext_end, self.total_steps).reshape(-1, 1).squeeze(axis=1)

    def update(self):
        """
        Get the next input for the CANN instance.
        """
        pos = self._step()
        if pos is False:
            return False

        self.current_step.value += 1
        self.inputs.value = self.cann_instance.get_stimulus_by_pos(pos)

class CustomTracking1D(Tracking1D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _make_Iext_sequence(self):
        ...

    def update(self):
        ...