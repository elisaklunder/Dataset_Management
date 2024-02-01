class Errors:
    def type_check(self, argument_name, argument, *expected_types):
        if type(argument) not in expected_types:
            types_str = " or ".join(str(type) for type in expected_types)
            raise TypeError(
                f"Invalid type for argument '{argument_name}'. Expected \
{types_str}, but got {type(argument)} instead."
            )

    def value_check(self, argument_name, argument, *accepted_arguments):
        if argument not in accepted_arguments:
            raise ValueError(
                f"Input '{argument}' for argument '{argument_name}' is not \
valid. Expected one of the following: {accepted_arguments}."
            )

    def ispositive(self, argument_name, argument):
        if argument <= 0:
            raise ValueError(
                f"'{argument_name}' should be a value grater than 0"
            )
