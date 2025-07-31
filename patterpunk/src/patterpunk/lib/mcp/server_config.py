from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MCPServerConfig:
    name: str
    url: Optional[str] = None
    command: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.url and not self.command:
            raise ValueError(
                "MCPServerConfig must specify either 'url' for HTTP transport or 'command' for stdio transport"
            )

        if self.url and self.command:
            raise ValueError(
                "MCPServerConfig cannot specify both 'url' and 'command' - choose one transport method"
            )

    @property
    def is_http_transport(self) -> bool:
        return self.url is not None

    @property
    def is_stdio_transport(self) -> bool:
        return self.command is not None
