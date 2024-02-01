from typing import Any, Callable, Tuple


class PreprocessingPipeline:
    def __init__(
        self, *preprocessing_techniques: Tuple[Callable, ...]
    ) -> None:
        self._preprocessing_steps = list(preprocessing_techniques)

    def __call__(self, data: Any) -> Any:
        return self._apply_pipeline(data)

    def _apply_pipeline(self, data: Any) -> Any:
        for step in self._preprocessing_steps:
            if not callable(step):
                raise TypeError(
                    f"Preprocessing step {step} is not a callable technique."
                )
            data = step(data)
        return data
