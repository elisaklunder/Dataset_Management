from typing import Any


class Errors:
    def type_check(
        self, argument_name: str, argument: Any, *expected_types: Any
    ) -> None:
        """
        Public method used through out many classes to perform type checks.

        Args:
            argument_name (str): name of the argument to be checked.

            argument (Any): argument to be checked for type.

            *expected_types (Any): One or more expected types.

        Raises:
            TypeError: If the type of the argument does not match any of the
            expected types. The error message includes details about the
            expected types and the actual type.
        """
        if not any(isinstance(argument, type) for type in expected_types):
            types_str = " or ".join(str(type) for type in expected_types)
            raise TypeError(
                f"Invalid type for argument '{argument_name}'. Expected \
{types_str}, but got {type(argument)} instead."
            )

    def value_check(
        self, argument_name: str, argument: Any, *accepted_arguments: Any
    ) -> None:
        """
        Public method used through out many classes to perform specific value
        checks.

        Args:
            argument_name (str): The name of the argument being checked.

            argument (Any): The value of the argument to be checked.

            *accepted_arguments (Any): One or more accepted values for the
            argument.


        Raises:
            ValueError: If the value of the argument is not among the accepted
            values.
        """

        if argument not in accepted_arguments:
            raise ValueError(
                f"Input '{argument}' for argument '{argument_name}' is not \
valid. Expected one of the following: {accepted_arguments}."
            )

    def ispositive(self, argument_name: str, argument: int):
        """
        Public method used through out many classes to check if the value of
        the given argument is positive

        Args:
            argument_name (str): The name of the argument tha should have a
            positive value.

            argument (int): The value of the argument.

        Raises:
            ValueError: if the value of the argument is not greater than 0.
        """
        if argument <= 0:
            raise ValueError(
                f"'{argument_name}' should be a value grater than 0"
            )
