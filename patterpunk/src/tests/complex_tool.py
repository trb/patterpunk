from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from patterpunk.lib.function_to_tool.converter import function_to_tool

PROMPT = (
    "Call record_contact exactly once with these arguments: "
    'names ["Ann", "Annie"], scores [3, 5], '
    'address {"street": "1 Main St", "city": "Halifax"}, '
    'previous_addresses [{"street": "2 Side Rd", "city": "Truro"}], '
    'priority "high", and no note. '
    "After the tool answers, reply with the contact identifier it returns."
)

EXPECTED_ARGUMENTS = {
    "names": ["Ann", "Annie"],
    "scores": [3, 5],
    "address": {"street": "1 Main St", "city": "Halifax"},
    "previous_addresses": [{"street": "2 Side Rd", "city": "Truro"}],
    "priority": "high",
}

CONTACT_IDENTIFIER = "rc-7"


class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"


class Address(BaseModel):
    street: str = Field(description="Street name and house number")
    city: str = Field(description="City name")


def make_record_contact_tool():
    recorded_calls = []

    def record_contact(
        names: list[str],
        scores: list[int],
        address: Address,
        previous_addresses: list[Address],
        priority: Priority,
        note: Optional[str] = None,
    ) -> str:
        """Record a contact in the address book.

        Args:
            names: Every name the contact uses
            scores: Integer ratings of the contact
            address: Current address
            previous_addresses: Earlier addresses, most recent first
            priority: Follow-up priority
            note: Free-text note, omitted when there is none
        """
        recorded_calls.append(
            {
                "names": names,
                "scores": scores,
                "address": address,
                "previous_addresses": previous_addresses,
                "priority": priority,
                "note": note,
            }
        )
        return f"Recorded contact {CONTACT_IDENTIFIER}."

    return record_contact, recorded_calls


def record_contact_tool_definition() -> dict:
    record_contact, _ = make_record_contact_tool()
    return function_to_tool(record_contact)


def assert_tool_received_expected_arguments(recorded_calls: list) -> None:
    assert recorded_calls, "The model never called record_contact"
    arguments = dict(recorded_calls[0])
    note = arguments.pop("note")
    assert arguments == EXPECTED_ARGUMENTS
    assert not note
