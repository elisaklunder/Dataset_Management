from typing import Any, Callable, Tuple


class PreprocessingPipeline:
    def __init__(
        self, *preprocessing_techniques: Tuple[Callable, ...]
    ) -> None:
        """
        Constructor of data-type agnostic class

        Args:
        *preprocessing_techniques (Tuple[Callable, ...]): One or more callable
        preprocessing techniques represented as functions. These techniques
        will be applied sequentially during the preprocessing.

        """
        self._preprocessing_steps = list(preprocessing_techniques)

    def __call__(self, data: Any) -> Any:
        """
        Magic method implementing the call functionality for the class

        Args:
            data (Any): The data to be processed.

        Returns:
            Any: The processed data after applying all the preprocessing steps.
        """
        return self._apply_pipeline(data)

    def _apply_pipeline(self, data: Any) -> Any:
        """_summary_

        Args:
            data (Any): data to be preprocessed.

        Raises:
            TypeError: if one of the given preprocessing tecniques is not a
            callable.

        Returns:
            Any: processed data.
        """
        for step in self._preprocessing_steps:
            if not callable(step):
                raise TypeError(
                    f"Preprocessing step {step} is not a callable technique."
                )
            data = step(data)
        return data
