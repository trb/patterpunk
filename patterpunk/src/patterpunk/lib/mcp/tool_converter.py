from typing import Any, Dict, List

from patterpunk.llm.types import ToolDefinition


def mcp_tools_to_patterpunk_tools(mcp_tools: List[Dict[str, Any]]) -> ToolDefinition:
    patterpunk_tools = []

    for mcp_tool in mcp_tools:
        patterpunk_tool = _convert_mcp_tool_to_patterpunk(mcp_tool)
        patterpunk_tools.append(patterpunk_tool)

    return patterpunk_tools


def _convert_mcp_tool_to_patterpunk(mcp_tool: Dict[str, Any]) -> Dict[str, Any]:
    name = mcp_tool.get("name", "")
    description = mcp_tool.get("description", "")
    input_schema = mcp_tool.get("inputSchema", {})

    parameters = _convert_mcp_schema_to_openai_parameters(input_schema)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "_mcp_server": mcp_tool.get("_mcp_server"),
        },
    }


def _convert_mcp_schema_to_openai_parameters(
    mcp_schema: Dict[str, Any],
) -> Dict[str, Any]:
    if not mcp_schema:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    parameters = {
        "type": mcp_schema.get("type", "object"),
        "properties": mcp_schema.get("properties", {}),
        "required": mcp_schema.get("required", []),
        "additionalProperties": False,
    }

    if "description" in mcp_schema:
        parameters["description"] = mcp_schema["description"]

    return parameters


def extract_mcp_server_from_tool_call(
    tool_call_name: str, available_tools: ToolDefinition
) -> str:
    for tool in available_tools:
        if tool.get("function", {}).get("name") == tool_call_name:
            return tool.get("function", {}).get("_mcp_server", "")
    return ""
