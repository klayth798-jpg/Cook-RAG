"""
MCP 客户端管理器
负责连接外部 MCP Server（如 mcp-server-fetch），提供统一工具调用接口。
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPCallResult:
    success: bool
    content: str
    raw_data: Any = None
    error: Optional[str] = None


class MCPSkillClient:
    KNOWN_SERVERS = {
        "fetch": {
            "command_options": [
                {"command": "uvx", "args": ["mcp-server-fetch"]},
                {"command": "python", "args": ["-m", "mcp_server_fetch"]},
                {"command": "python3", "args": ["-m", "mcp_server_fetch"]},
            ],
            "description": "官方网页抓取 MCP Server",
        }
    }

    def __init__(self):
        self._resolved_commands: Dict[str, dict] = {}

    def _resolve_server_command(self, server_name: str) -> Optional[dict]:
        if server_name in self._resolved_commands:
            return self._resolved_commands[server_name]
        server_config = self.KNOWN_SERVERS.get(server_name)
        if not server_config:
            return None
        for option in server_config["command_options"]:
            if shutil.which(option["command"]):
                self._resolved_commands[server_name] = option
                return option
        return None

    def is_server_available(self, server_name: str) -> bool:
        return self._resolve_server_command(server_name) is not None

    def list_tools(self, server_name: str) -> List[MCPToolInfo]:
        try:
            return asyncio.run(self._async_list_tools(server_name))
        except Exception as error:
            logger.error(f"列出 MCP 工具失败 ({server_name}): {error}")
            return []

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        try:
            return asyncio.run(self._async_call_tool(server_name, tool_name, arguments))
        except Exception as error:
            logger.error(f"MCP 工具调用失败 ({server_name}/{tool_name}): {error}")
            return MCPCallResult(success=False, content=f"MCP 工具调用失败: {error}", error=str(error))

    async def _async_list_tools(self, server_name: str) -> List[MCPToolInfo]:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        resolved = self._resolve_server_command(server_name)
        if not resolved:
            return []

        server_params = StdioServerParameters(command=resolved["command"], args=resolved["args"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                return [
                    MCPToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=getattr(tool, "inputSchema", {}) or {},
                    )
                    for tool in tools_response.tools
                ]

    async def _async_call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        resolved = self._resolve_server_command(server_name)
        if not resolved:
            return MCPCallResult(success=False, content=f"MCP Server '{server_name}' 不可用", error="Server unavailable")

        server_params = StdioServerParameters(command=resolved["command"], args=resolved["args"])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                logger.info(f"🔧 调用 MCP 工具: {server_name}/{tool_name}, 参数: {json.dumps(arguments, ensure_ascii=False)[:200]}")
                result = await session.call_tool(tool_name, arguments)
                parts = []
                for item in result.content:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(str(item.data))
                combined = "\n".join(parts)
                if getattr(result, "isError", False):
                    return MCPCallResult(success=False, content=combined, raw_data=result, error=combined)
                return MCPCallResult(success=True, content=combined, raw_data=result)


def check_mcp_environment() -> Dict[str, bool]:
    client = MCPSkillClient()
    results = {name: client.is_server_available(name) for name in MCPSkillClient.KNOWN_SERVERS}
    try:
        import mcp  # noqa: F401
        results["mcp_sdk"] = True
    except ImportError:
        results["mcp_sdk"] = False
    return results
