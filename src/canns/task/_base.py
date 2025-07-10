from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Task(ABC):
    """
    A generic Task Abstract Base Class (ABC) designed to standardize the data
    handling workflow for various AI tasks.

    This class defines a standard interface that any concrete task class
    inheriting from it must implement, covering the core methods for data

    acquisition, processing, visualization, and access. It also provides a
    universal data-saving functionality.

    Attributes:
        data: A container for the loaded or generated data, typically a
              NumPy array or a dictionary of arrays.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the Task instance.

        Args:
            config (dict, optional): A dictionary containing configurations
                                     for the task. Defaults to None.
        """
        self.data = None  # Data is not loaded at initialization

    @abstractmethod
    def get_data(self) -> None:
        """
        Abstract core method for data acquisition.

        Subclasses must implement this method. Depending on the task type,
        the implementation could be:
        - Downloading and loading data from the web.
        - Reading data from the local filesystem.
        - Generating synthetic data in real-time.

        After execution, this method should assign the processed data to `self.data`.
        """
        pass

    def save_data(self, filepath: str) -> None:
        """
        Saves the task data to a compressed `.npz` file.

        This is a concrete method that all subclasses can use directly. It
        assumes `self.data` is either a dictionary or a NumPy array.

        Args:
            filepath (str): The path to save the file, which should end
                            with `.npz`.

        Raises:
            ValueError: If `self.data` is None, as there is nothing to save.
        """
        if self.data is None:
            raise ValueError(
                "Data has not been loaded or generated (self.data is None). Please call get_data() first."
            )

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(self.data, dict):
            np.savez_compressed(output_path, **self.data)
        else:
            np.savez_compressed(output_path, data=self.data)

        print(f"Data successfully saved to: {output_path}")

    def load_data(self, filepath: str) -> None:
        """
        Loads data from a compressed `.npz` file.

        This is a concrete method that all subclasses can use directly. It
        assumes the file contains either a dictionary or a single NumPy array.

        Args:
            filepath (str): The path to the file to load, which should end
                            with `.npz`.

        Raises:
            ValueError: If the file does not exist or cannot be loaded.
        """
        output_path = Path(filepath)
        if not output_path.exists():
            raise ValueError(f"File {output_path} does not exist.")

        loaded_data = np.load(output_path, allow_pickle=True)
        if len(loaded_data.files) == 1:
            self.data = loaded_data[loaded_data.files[0]]
        else:
            self.data = {key: loaded_data[key] for key in loaded_data.files}

        print(f"Data successfully loaded from: {output_path}")

    @abstractmethod
    def show_data(
        self,
        show=True,
        save_path=None,
    ) -> None:
        """
        Abstract method to display a task.

        Subclasses must implement this to visualize a sample in a way that is
        appropriate for its data type (e.g., plotting an image, a waveform,
        or printing text).
        """
        pass
