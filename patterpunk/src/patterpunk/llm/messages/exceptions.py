class BadParameterError(Exception):
    pass


class UnexpectedFunctionCallError(Exception):
    pass


class StructuredOutputNotPydanticLikeError(Exception):
    pass


class StructuredOutputFailedToParseError(Exception):
    pass
