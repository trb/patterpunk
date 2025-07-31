from typing import List, TypedDict, Dict, Optional


class ToolFunctionParameters(TypedDict, total=False):
    type: str
    properties: Dict[str, Dict[str, str]]
    required: List[str]
    additionalProperties: bool


class ToolFunction(TypedDict):
    name: str
    description: str
    parameters: ToolFunctionParameters
    strict: Optional[bool]


class Tool(TypedDict):
    type: str
    function: ToolFunction


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: ToolCallFunction


ToolDefinition = List[Tool]
ToolCallList = List[ToolCall]
