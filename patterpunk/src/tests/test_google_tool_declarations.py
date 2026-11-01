from enum import Enum
from typing import Optional
from unittest.mock import Mock

from pydantic import BaseModel, Field

from patterpunk.lib.function_to_tool.converter import function_to_tool
from patterpunk.llm.models.google import GoogleModel


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


class Address(BaseModel):
    street: str = Field(description="Street and house number")
    city: str


def get_fact_details(fact_uuids: list[str]) -> str:
    """Retrieve full content for specific facts.

    Args:
        fact_uuids: List of fact UUIDs to retrieve
    """
    return ""


def add_scores(scores: list[int]) -> str:
    """Store integer scores.

    Args:
        scores: Scores between 1 and 10
    """
    return ""


def record_addresses(address: Address, previous_addresses: list[Address]) -> str:
    """Record where a contact lives and lived.

    Args:
        address: Current address
        previous_addresses: Earlier addresses
    """
    return ""


def set_priority(priority: Priority) -> str:
    """Set the follow-up priority.

    Args:
        priority: How urgent the follow-up is
    """
    return ""


def search_facts(query: str, limit: Optional[int] = None) -> str:
    """Search facts.

    Args:
        query: Search terms
        limit: Maximum number of results
    """
    return ""


def declaration_for(func):
    model = GoogleModel(model="gemini-2.5-pro", client=Mock())
    google_tools = model._convert_tools_to_google_format([function_to_tool(func)])
    return google_tools[0].function_declarations[0]


def test_declaration_sends_the_unchanged_json_schema_only():
    declaration = declaration_for(get_fact_details)

    assert declaration.name == "get_fact_details"
    assert declaration.description == "Retrieve full content for specific facts."
    assert (
        declaration.parameters_json_schema
        == function_to_tool(get_fact_details)["function"]["parameters"]
    )
    assert declaration.parameters is None


def test_list_of_strings_keeps_item_type_and_description():
    schema = declaration_for(get_fact_details).parameters_json_schema

    fact_uuids = schema["properties"]["fact_uuids"]
    assert fact_uuids["type"] == "array"
    assert fact_uuids["items"] == {"type": "string"}
    assert fact_uuids["description"] == "List of fact UUIDs to retrieve"
    assert schema["required"] == ["fact_uuids"]


def test_list_of_integers_keeps_item_type():
    schema = declaration_for(add_scores).parameters_json_schema

    scores = schema["properties"]["scores"]
    assert scores["type"] == "array"
    assert scores["items"] == {"type": "integer"}


def test_nested_model_keeps_properties_required_and_descriptions():
    schema = declaration_for(record_addresses).parameters_json_schema

    assert schema["properties"]["address"]["$ref"] == "#/$defs/Address"
    assert schema["properties"]["address"]["description"] == "Current address"
    assert schema["properties"]["previous_addresses"]["items"] == {
        "$ref": "#/$defs/Address"
    }
    address_definition = schema["$defs"]["Address"]
    assert address_definition["type"] == "object"
    assert address_definition["required"] == ["street", "city"]
    assert (
        address_definition["properties"]["street"]["description"]
        == "Street and house number"
    )
    assert address_definition["properties"]["city"]["type"] == "string"


def test_enum_keeps_allowed_values():
    schema = declaration_for(set_priority).parameters_json_schema

    assert schema["properties"]["priority"]["$ref"] == "#/$defs/Priority"
    assert schema["properties"]["priority"]["description"] == (
        "How urgent the follow-up is"
    )
    assert schema["$defs"]["Priority"]["enum"] == ["low", "high"]


def test_optional_parameter_is_nullable_and_not_required():
    schema = declaration_for(search_facts).parameters_json_schema

    limit = schema["properties"]["limit"]
    assert {"type": "integer"} in limit["anyOf"]
    assert {"type": "null"} in limit["anyOf"]
    assert limit["description"] == "Maximum number of results"
    assert schema["required"] == ["query"]
