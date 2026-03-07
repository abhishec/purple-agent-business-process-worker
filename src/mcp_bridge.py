from __future__ import annotations
import json
import httpx
from src.config import TOOL_TIMEOUT

# Per-endpoint format cache: "jsonrpc" | "simple"
# Populated on first successful tool discovery to avoid double-requests on every call.
_endpoint_format: dict[str, str] = {}


def validate_tool_call(
    tool_name: str,
    params: dict,
    tools_list: list[dict],
) -> tuple[bool, str]:
    """Pre-flight: verify tool exists and required params are present.

    Returns (valid, error_msg). If tools_list is empty we cannot validate —
    allow through rather than blocking legitimate calls.
    """
    if not tools_list:
        return True, ""  # can't validate without schema — allow through

    tool_schema = next((t for t in tools_list if t.get("name") == tool_name), None)
    if tool_schema is None:
        # Tool not in discovered list — likely hallucinated
        available = [t.get("name") for t in tools_list[:10]]
        return False, f"Tool '{tool_name}' not in available tools. Available: {available}"

    # Check required params — handle both input_schema (Anthropic) and inputSchema (MCP spec)
    input_schema = tool_schema.get("input_schema") or tool_schema.get("inputSchema", {})
    required = input_schema.get("required", [])
    missing = [r for r in required if r not in params]
    if missing:
        return False, f"Tool '{tool_name}' missing required params: {missing}"

    return True, ""


def _is_empty_result(result: dict) -> bool:
    """Return True when the tool result carries no useful data."""
    if "error" in result:
        return False  # errors are meaningful — not "empty"
    for key in ("data", "result", "items", "records", "rows"):
        val = result.get(key)
        if val is not None:
            if isinstance(val, (list, dict)) and len(val) == 0:
                continue  # empty container — keep checking other keys
            return False  # non-empty value found
    # If none of the expected keys had content, consider it empty
    return all(
        (result.get(k) is None or result.get(k) == [] or result.get(k) == {})
        for k in ("data", "result", "items", "records", "rows")
    )


def _format_tools(tools: list[dict]) -> list[dict]:
    """Normalize tool schemas to Anthropic format."""
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("inputSchema") or t.get("input_schema") or {
                "type": "object", "properties": {}
            },
        }
        for t in tools
    ]


async def discover_tools(tools_endpoint: str, session_id: str = "") -> list[dict]:
    """Discover tools — tries JSON-RPC 2.0 first, then simple GET format.

    Two MCP server formats are supported:
    - JSON-RPC 2.0: POST /mcp with {"jsonrpc":"2.0","method":"tools/list",...}  (tau2-bench)
    - Simple:       GET /mcp/tools?session_id=...                                (agent-bench)

    The detected format is cached per endpoint for use by call_tool().
    """
    ep = tools_endpoint

    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        # ── Format 1: JSON-RPC 2.0 (tau2-bench / standard MCP) ──────────────
        if _endpoint_format.get(ep) != "simple":
            url = f"{ep}/mcp"
            if session_id:
                url = f"{url}?session_id={session_id}"
            try:
                resp = await client.post(url, json={
                    "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
                })
                data = resp.json()
                tools = data.get("result", {}).get("tools", [])
                if tools:
                    _endpoint_format[ep] = "jsonrpc"
                    print(f"[mcp] {ep}: using JSON-RPC format, {len(tools)} tools", flush=True)
                    return _format_tools(tools)
            except Exception as e:
                print(f"[mcp] JSON-RPC discover failed for {ep}: {e}", flush=True)

        # ── Format 2: Simple GET /mcp/tools (agent-bench) ────────────────────
        url2 = f"{ep}/mcp/tools"
        if session_id:
            url2 = f"{url2}?session_id={session_id}"
        try:
            resp2 = await client.get(url2)
            data2 = resp2.json()
            tools2 = data2 if isinstance(data2, list) else data2.get("tools", [])
            if tools2:
                _endpoint_format[ep] = "simple"
                print(f"[mcp] {ep}: using simple format, {len(tools2)} tools", flush=True)
                return _format_tools(tools2)
        except Exception as e:
            print(f"[mcp] simple GET discover failed for {ep}: {e}", flush=True)

    print(f"[mcp] discover_tools: no tools found at {ep}", flush=True)
    return []


def _parse_jsonrpc_result(data: dict) -> dict:
    """Parse a JSON-RPC tool call response into a plain dict."""
    if "error" in data:
        err = data["error"]
        return {"error": err.get("message", str(err)) if isinstance(err, dict) else str(err)}

    result = data.get("result", {})

    if result.get("isError"):
        content = result.get("content", [])
        error_text = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
        return {"error": error_text or "Tool returned an error"}

    content = result.get("content", [])
    if not content:
        return result

    if len(content) == 1 and content[0].get("type") == "text":
        text = content[0]["text"]
        try:
            return json.loads(text)
        except Exception:
            return {"result": text}

    return {"content": content}


async def call_tool(
    tools_endpoint: str,
    tool_name: str,
    params: dict,
    session_id: str,
    tools_list: list[dict] | None = None,
) -> dict:
    """Call a tool — uses the format detected during discover_tools().

    Two MCP server formats:
    - JSON-RPC 2.0: POST /mcp with {"jsonrpc":"2.0","method":"tools/call",...}  (tau2-bench)
    - Simple:       POST /mcp with {"tool":"name","params":{...},"session_id":...} (agent-bench)
    """
    # Pre-flight validation
    valid, error_msg = validate_tool_call(tool_name, params, tools_list or [])
    if not valid:
        return {"error": error_msg, "validation_failed": True}

    ep = tools_endpoint
    fmt = _endpoint_format.get(ep, "jsonrpc")  # default to jsonrpc (tau2-bench)

    async with httpx.AsyncClient(timeout=TOOL_TIMEOUT) as client:
        if fmt == "simple":
            # ── Agent-bench simple format: POST /mcp with {tool, params, session_id} ──
            url = f"{ep}/mcp"
            resp = await client.post(url, json={
                "tool": tool_name,
                "params": params,
                "session_id": session_id,
            })
            resp.raise_for_status()
            return resp.json()
        else:
            # ── JSON-RPC 2.0 format: POST /mcp?session_id=... with jsonrpc payload ──
            url = f"{ep}/mcp"
            if session_id:
                url = f"{url}?session_id={session_id}"
            resp = await client.post(url, json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
            })
            resp.raise_for_status()
            data = resp.json()

            # If server returned a validation error (wrong format), try simple format
            if "detail" in data and "result" not in data:
                print(f"[mcp] JSON-RPC call rejected, retrying with simple format", flush=True)
                _endpoint_format[ep] = "simple"
                url2 = f"{ep}/mcp"
                resp2 = await client.post(url2, json={
                    "tool": tool_name,
                    "params": params,
                    "session_id": session_id,
                })
                resp2.raise_for_status()
                return resp2.json()

            return _parse_jsonrpc_result(data)
