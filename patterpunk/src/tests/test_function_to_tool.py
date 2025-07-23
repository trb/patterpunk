# ai generated open-hands
import pytest
from typing import Optional, List, Dict, Union
from patterpunk.lib.function_to_tool import function_to_tool, functions_to_tools


def test_simple_function():

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    tool = function_to_tool(add_numbers)

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "add_numbers"
    assert tool["function"]["description"] == "Add two numbers together."

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    assert set(params["required"]) == {"a", "b"}

    assert params["properties"]["a"]["type"] == "integer"
    assert params["properties"]["b"]["type"] == "integer"


def test_function_with_optional_parameters():

    def greet(name: str, greeting: str = "Hello") -> str:

        return f"{greeting}, {name}!"

    tool = function_to_tool(greet)

    params = tool["function"]["parameters"]
    assert params["required"] == ["name"]
    assert "greeting" not in params["required"]
    assert params["properties"]["greeting"]["default"] == "Hello"


def test_function_with_various_types():

    def process_data(
        text: str,
        count: int,
        ratio: float,
        enabled: bool,
        items: List[str],
        mapping: Dict[str, int],
    ) -> dict:

        return {}

    tool = function_to_tool(process_data)

    props = tool["function"]["parameters"]["properties"]
    assert props["text"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["ratio"]["type"] == "number"
    assert props["enabled"]["type"] == "boolean"
    assert props["items"]["type"] == "array"
    assert props["items"]["items"]["type"] == "string"
    assert props["mapping"]["type"] == "object"


def test_function_with_optional_type():

    def search(query: str, limit: Optional[int] = None) -> List[str]:

        return []

    tool = function_to_tool(search)

    params = tool["function"]["parameters"]
    assert params["required"] == ["query"]

    limit_prop = params["properties"]["limit"]
    assert "anyOf" in limit_prop
    assert limit_prop["default"] is None


def test_function_with_union_type():

    def convert(value: Union[str, int]) -> str:

        return str(value)

    tool = function_to_tool(convert)

    value_prop = tool["function"]["parameters"]["properties"]["value"]
    assert "anyOf" in value_prop or "type" in value_prop


def test_function_without_annotations():

    def legacy_func(param):

        return param

    tool = function_to_tool(legacy_func)

    params = tool["function"]["parameters"]
    assert params["properties"]["param"]["type"] == "string"


def test_function_with_google_style_docstring():

    def calculate_area(length: float, width: float, unit: str = "m") -> float:
        """Calculate the area of a rectangle.

        This function computes the area by multiplying length and width.

        Args:
            length: The length of the rectangle in specified units
            width: The width of the rectangle in specified units
            unit: The unit of measurement (default: meters)

        Returns:
            The area of the rectangle
        """
        return length * width

    tool = function_to_tool(calculate_area)

    desc = tool["function"]["description"]
    assert "Calculate the area of a rectangle." in desc
    assert "This function computes the area" in desc
    assert "Args:" not in desc
    assert "Returns:" not in desc

    props = tool["function"]["parameters"]["properties"]
    assert "description" in props["length"]
    assert "rectangle" in props["length"]["description"]
    assert "description" in props["width"]
    assert "description" in props["unit"]
    assert "meters" in props["unit"]["description"]


def test_function_without_docstring():

    def no_docs(param: str) -> str:
        return param

    tool = function_to_tool(no_docs)

    assert tool["function"]["description"] == "no_docs"
    assert "description" not in tool["function"]["parameters"]["properties"]["param"]


def test_functions_to_tools():

    def func1(x: int) -> int:

        return x * 2

    def func2(y: str) -> str:

        return y.upper()

    tools = functions_to_tools([func1, func2])

    assert len(tools) == 2
    assert tools[0]["function"]["name"] == "func1"
    assert tools[1]["function"]["name"] == "func2"


def test_function_with_no_parameters():

    def get_timestamp() -> str:

        return "2023-01-01"

    tool = function_to_tool(get_timestamp)

    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["required"] == []


def test_function_with_args_kwargs():

    def flexible_func(required: str, *args, **kwargs) -> str:

        return required

    tool = function_to_tool(flexible_func)

    params = tool["function"]["parameters"]
    assert list(params["properties"].keys()) == ["required"]
    assert params["required"] == ["required"]


def test_function_with_multiline_param_description():

    def complex_function(data: str) -> dict:
        """Process complex data.

        Args:
            data: A complex data string that might contain
                  multiple lines of information and needs
                  detailed explanation of its format
        """
        return {}

    tool = function_to_tool(complex_function)

    data_desc = tool["function"]["parameters"]["properties"]["data"]["description"]
    assert "complex data string" in data_desc
    assert "multiple lines" in data_desc

    assert "\n" not in data_desc


def test_invalid_function():

    with pytest.raises(ValueError):
        function_to_tool("not a function")


def test_docstring_without_args_section():

    def simple_func(param: str) -> str:
        """This is a simple function that does something useful.

        It processes the input parameter in some way.
        """
        return param

    tool = function_to_tool(simple_func)

    desc = tool["function"]["description"]
    assert "simple function" in desc
    assert "processes the input" in desc

    param_prop = tool["function"]["parameters"]["properties"]["param"]
    assert "description" not in param_prop


def test_function_with_complex_types():

    def process_nested(
        items: List[Dict[str, Union[str, int]]],
        config: Optional[Dict[str, bool]] = None,
    ) -> bool:

        return True

    tool = function_to_tool(process_nested)

    params = tool["function"]["parameters"]
    assert "items" in params["properties"]
    assert "config" in params["properties"]
    assert params["required"] == ["items"]


def test_function_with_default_none():

    def optional_param(value: Optional[str] = None) -> str:

        return value or "default"

    tool = function_to_tool(optional_param)

    params = tool["function"]["parameters"]
    assert params["required"] == []
    assert params["properties"]["value"]["default"] is None
