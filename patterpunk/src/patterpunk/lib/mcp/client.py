import json
import subprocess
from typing import Any, Dict, List, Optional, Union

from .exceptions import MCPConnectionError, MCPDependencyError, MCPRequestError
from .server_config import MCPServerConfig

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class MCPClient:
    def __init__(self, server_configs: List[MCPServerConfig]) -> None:
        self._configs = server_configs
        self._sessions: Dict[str, str] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._connected = False

    def connect_all(self) -> None:
        if self._connected:
            return

        for config in self._configs:
            try:
                if config.is_http_transport:
                    self._connect_http_server(config)
                elif config.is_stdio_transport:
                    self._connect_stdio_server(config)
            except Exception as e:
                raise MCPConnectionError(config.name, str(e))

        self._connected = True

    def disconnect_all(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)

        self._processes.clear()
        self._sessions.clear()
        self._connected = False

    def get_available_tools(self) -> List[Dict[str, Any]]:
        self._ensure_connected()
        all_tools = []

        for config in self._configs:
            try:
                response = self._send_request(
                    config, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
                )

                if "result" in response and "tools" in response["result"]:
                    tools = response["result"]["tools"]
                    for tool in tools:
                        tool["_mcp_server"] = config.name
                    all_tools.extend(tools)

            except Exception as e:
                raise MCPRequestError(config.name, "tools/list", str(e))

        return all_tools

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
    ) -> Any:
        self._ensure_connected()

        config = self._find_server_for_tool(tool_name, server_name)
        if not config:
            raise MCPRequestError(
                server_name or "unknown",
                "tools/call",
                f"Tool '{tool_name}' not found on any connected MCP server",
            )

        try:
            response = self._send_request(
                config,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
            )

            if "error" in response:
                error = response["error"]
                raise MCPRequestError(
                    config.name, "tools/call", f"Tool execution failed: {error}"
                )

            return response.get("result")

        except Exception as e:
            raise MCPRequestError(config.name, "tools/call", str(e))

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect_all()

    def _connect_http_server(self, config: MCPServerConfig) -> None:
        if not HAS_REQUESTS:
            raise MCPDependencyError("requests")

        try:
            response = self._send_http_request(
                config.url,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "patterpunk", "version": "1.0.0"},
                    },
                },
                timeout=config.timeout,
            )

            if "error" in response:
                raise Exception(f"Initialize failed: {response['error']}")

            session_id = response.get("result", {}).get("sessionId", "")
            self._sessions[config.name] = session_id

        except requests.RequestException as e:
            raise MCPConnectionError(config.name, f"HTTP connection failed: {e}")

    def _connect_stdio_server(self, config: MCPServerConfig) -> None:
        try:
            process = subprocess.Popen(
                config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=config.env,
            )

            self._processes[config.name] = process

            response = self._send_stdio_request(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "patterpunk", "version": "1.0.0"},
                    },
                },
            )

            if "error" in response:
                raise Exception(f"Initialize failed: {response['error']}")

        except subprocess.SubprocessError as e:
            raise MCPConnectionError(config.name, f"Stdio connection failed: {e}")

    def _send_request(
        self, config: MCPServerConfig, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        if config.is_http_transport:
            return self._send_http_request(
                config.url,
                payload,
                session_id=self._sessions.get(config.name),
                timeout=config.timeout,
            )
        elif config.is_stdio_transport:
            process = self._processes[config.name]
            return self._send_stdio_request(process, payload)
        else:
            raise ValueError(f"Unknown transport type for server {config.name}")

    def _send_http_request(
        self,
        url: str,
        payload: Dict[str, Any],
        session_id: Optional[str] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        if not HAS_REQUESTS:
            raise MCPDependencyError("requests")

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _send_stdio_request(
        self, process: subprocess.Popen, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        request_line = json.dumps(payload) + "\n"
        process.stdin.write(request_line)
        process.stdin.flush()

        response_line = process.stdout.readline()
        if not response_line:
            raise Exception("No response from MCP server")

        return json.loads(response_line.strip())

    def _find_server_for_tool(
        self, tool_name: str, preferred_server: Optional[str] = None
    ) -> Optional[MCPServerConfig]:
        if preferred_server:
            return next((c for c in self._configs if c.name == preferred_server), None)

        tools = self.get_available_tools()
        for tool in tools:
            if tool.get("name") == tool_name:
                server_name = tool.get("_mcp_server")
                return next((c for c in self._configs if c.name == server_name), None)

        return None

    def __enter__(self):
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect_all()
