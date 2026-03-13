from __future__ import annotations
import asyncio
import json
from typing import Callable, Awaitable

import httpx

from src.config import BRAINOS_API_URL, BRAINOS_API_KEY, BRAINOS_ORG_ID, TASK_TIMEOUT


class BrainOSUnavailableError(Exception):
    pass


async def run_task(
    message: str,
    system_context: str,
    on_tool_call: Callable[[str, dict], Awaitable[dict]],
    session_id: str,
    organization_id: str | None = None,
    worker_id: str | None = None,
    eval_hints: dict | None = None,
    timeout: float | None = None,
) -> str:
    """
    Stream a task through BrainOS copilot API.
    Handles SSE stream, detects tool_call events, calls on_tool_call, injects results.
    Raises BrainOSUnavailableError on connection failure or timeout.

    eval_hints: passed as evalHints to BrainOS — enables local LLM routing,
                bandit strategy selection, and prescreen (cheap inference path).
                Example: {"taskType": "crm", "model": "claude-haiku-4-5-20251001",
                          "preScreenEnabled": True, "preScreenConfidence": 0.65,
                          "banditEnabled": True, "banditCategory": "crm_analytics"}
    """
    if not BRAINOS_API_KEY or not BRAINOS_ORG_ID:
        raise BrainOSUnavailableError("BrainOS credentials not configured")

    org_id = organization_id or BRAINOS_ORG_ID
    url = f"{BRAINOS_API_URL}/api/copilot/chat"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        # Support both cookie-based auth (eval runs) and API key auth (agent)
        **({"X-API-Key": BRAINOS_API_KEY} if BRAINOS_API_KEY else {}),
    }
    payload: dict = {
        "message": message,
        "conversationId": session_id,
        "organizationId": org_id,
        "isResearchMode": True,
        "extraParams": {"systemContext": system_context},
    }
    if worker_id:
        payload["workerId"] = worker_id
    if eval_hints:
        payload["evalHints"] = eval_hints

    effective_timeout = timeout or TASK_TIMEOUT

    async def _drain_sse(stream_resp) -> tuple[str, list[dict]]:
        final_answer = ""
        tool_results: list[dict] = []
        async for line in stream_resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if "tool_call" in event:
                tc = event["tool_call"]
                tool_result = await on_tool_call(tc.get("name", ""), tc.get("params", {}))
                tool_results.append({"name": tc.get("name", ""), "result": tool_result})
            if "answer" in event:
                final_answer = event["answer"]
            elif "text" in event:
                final_answer += event["text"]
        return final_answer, tool_results

    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code >= 400:
                    raise BrainOSUnavailableError(f"BrainOS returned {resp.status_code}")

                final_answer, tool_results = await _drain_sse(resp)

                # If tool results were collected but no final answer, send a follow-up
                if tool_results and not final_answer:
                    followup_payload = {
                        "message": "Tool results:",
                        "conversationId": session_id,
                        "organizationId": org_id,
                        "toolResults": tool_results,
                    }
                    async with client.stream("POST", url, headers=headers, json=followup_payload) as fu_resp:
                        if fu_resp.status_code >= 400:
                            raise BrainOSUnavailableError(f"BrainOS follow-up returned {fu_resp.status_code}")
                        final_answer, _ = await _drain_sse(fu_resp)

                return final_answer

    except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
        raise BrainOSUnavailableError(str(e)) from e


async def run_crm_task(
    prompt: str,
    crm_context: str,
    category: str,
    session_id: str,
    timeout: float = 35.0,
) -> str | None:
    """
    Route a CRM task through BrainOS local LLM (DeepSeek-R1-32B-AWQ).
    Uses prescreen + bandit to hit the cheap local LLM path first.
    Returns None if BrainOS is unavailable or returns an empty/refusal answer.

    Cost: ~$0 if local LLM handles it, ~$0.003 if Haiku fallback.
    """
    if not BRAINOS_API_KEY or not BRAINOS_ORG_ID:
        return None

    # Build a self-contained message: CRM context embedded directly so BrainOS
    # local LLM (DeepSeek-R1) can reason without web search or tool calls.
    message = (
        f"CRM Task — Category: {category}\n\n"
        f"Question: {prompt}\n\n"
        f"CRM Data:\n{crm_context[:20000]}"
    )
    system_context = (
        "You are a CRM data analyst. Answer the CRM question from the provided data. "
        "Return ONLY the exact answer value — no explanation, no prefix, no punctuation. "
        "If the answer is None/not found: respond with exactly: None"
    )
    eval_hints = {
        "taskType": "crm",
        "model": "claude-haiku-4-5-20251001",    # Haiku fallback if local LLM can't handle it
        "preScreenEnabled": True,
        "preScreenConfidence": 0.60,             # lower threshold — accept local LLM at 60%
        "banditEnabled": True,
        "banditCategory": f"crm_{category}",
        "evalPipeline": {
            "name": "crmarena_v2_brainos",
            "localLLMFirst": True,               # hint: try local LLM before Claude
        },
    }

    async def _noop_tool(name: str, params: dict) -> dict:
        return {"error": "no tools available in CRM direct mode"}

    try:
        answer = await asyncio.wait_for(
            run_task(
                message=message,
                system_context=system_context,
                on_tool_call=_noop_tool,
                session_id=session_id,
                eval_hints=eval_hints,
                timeout=timeout,
            ),
            timeout=timeout + 2.0,
        )
        # Validate: non-empty, not a refusal, not an HTML error page
        if not answer or len(answer.strip()) < 1:
            return None
        if answer.strip().startswith("<!"):
            return None
        return answer.strip()
    except (BrainOSUnavailableError, asyncio.TimeoutError) as e:
        print(f"[brainos-crm] unavailable cat={category}: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[brainos-crm] error cat={category}: {e}", flush=True)
        return None
