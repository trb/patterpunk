class MCPError(Exception):
    pass


class MCPConnectionError(MCPError):
    def __init__(self, server_name: str, message: str):
        self.server_name = server_name
        super().__init__(f"MCP connection error for server '{server_name}': {message}")


class MCPRequestError(MCPError):
    def __init__(self, server_name: str, method: str, message: str):
        self.server_name = server_name
        self.method = method
        super().__init__(
            f"MCP request error for server '{server_name}' method '{method}': {message}"
        )


class MCPDependencyError(MCPError):
    def __init__(self, dependency: str):
        super().__init__(
            f"Missing required dependency '{dependency}' for MCP functionality. "
            f"Install with: pip install {dependency}"
        )
