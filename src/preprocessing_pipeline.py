class PreprocessingPipeline:
    def __init__(self, *preprocessing_techniques):
        self._preprocessing_steps = list(preprocessing_techniques)

    def __call__(self, data):
        return self._apply_pipeline(data)

    def _apply_pipeline(self, data):
        for step in self._preprocessing_steps:
            if not callable(step):
                raise TypeError(
                    f"Preprocessing step {step} is not a callable technique."
                )
            data = step(data)
        return data
