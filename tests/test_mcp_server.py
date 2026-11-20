"""The MCP server, exercised through a real MCP client connected in-process."""
import asyncio
import json

import pytest

mcp = pytest.importorskip("mcp")   # optional dependency; skipped where it isn't installed


def _call(coro_fn):
    import mcp_server
    from mcp import Client

    async def main():
        async with Client(mcp_server.server) as client:
            return await coro_fn(client)
    return asyncio.run(main())


def _text(result):
    return "".join(getattr(c, "text", "") for c in result.content)


def test_lists_tools_without_admin_agents():
    tools = _call(lambda c: c.list_tools())
    names = {t.name for t in tools.tools}
    assert {"list_agents", "run_agent", "find_patients", "propose_followup"} <= names
    agents_listed = json.loads(_text(_call(lambda c: c.call_tool("list_agents", {}))))
    keys = {a["key"] for a in agents_listed}
    assert "retraining" not in keys and "imaging" not in keys and "patient_intelligence" in keys


def test_read_tool_and_proposal_only_write():
    import db
    found = json.loads(_text(_call(lambda c: c.call_tool("find_patients", {"query": "Ramesh"}))))
    found = found if isinstance(found, list) else [found]
    assert found[0]["id"] == 1
    before = db.connect().execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0]
    res = json.loads(_text(_call(lambda c: c.call_tool(
        "propose_followup", {"patient_id": 1, "reason": "review #46"}))))
    assert res["status"] == "awaiting human approval"
    conn = db.connect()
    row = conn.execute("SELECT status FROM agent_actions ORDER BY id DESC LIMIT 1").fetchone()
    assert conn.execute("SELECT COUNT(*) FROM agent_actions").fetchone()[0] == before + 1
    assert row[0] == "proposed"          # nothing executed — a clinician must approve


def test_run_agent_refuses_admin_only():
    res = _call(lambda c: c.call_tool("run_agent", {"key": "retraining", "params": {}}))
    assert res.is_error and "not available" in _text(res)
