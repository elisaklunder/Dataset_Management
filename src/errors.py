from typing import Any


class Errors:
    def type_check(
        self, argument_name: str, argument: Any, *expected_types: Any
    ) -> None:
        if not any(isinstance(argument, type) for type in expected_types):
            types_str = " or ".join(str(type) for type in expected_types)
            raise TypeError(
                f"Invalid type for argument '{argument_name}'. Expected \
{types_str}, but got {type(argument)} instead."
            )

    def value_check(
        self, argument_name: str, argument: Any, *accepted_arguments: Any
    ) -> None:
        if argument not in accepted_arguments:
            raise ValueError(
                f"Input '{argument}' for argument '{argument_name}' is not \
valid. Expected one of the following: {accepted_arguments}."
            )

    def ispositive(self, argument_name: str, argument: int):
        if argument <= 0:
            raise ValueError(
                f"'{argument_name}' should be a value grater than 0"
            )
