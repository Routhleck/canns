from __future__ import annotations

from canns.trainer import Trainer


class DummyTrainer(Trainer):
    def __init__(self) -> None:
        super().__init__(model=None)
        self.calls: list = []

    def train(self, train_data) -> None:
        self.calls.append(list(train_data))

    def predict(self, pattern, *args, **kwargs):
        return pattern


def test_configure_progress_updates_flags() -> None:
    trainer = DummyTrainer()
    trainer.configure_progress(show_iteration_progress=True, compiled_prediction=False)

    assert trainer.show_iteration_progress is True
    assert trainer.compiled_prediction is False


def test_predict_batch_delegates_to_predict() -> None:
    trainer = DummyTrainer()
    patterns = [1, 2, 3]

    result = trainer.predict_batch(patterns)

    assert result == patterns
