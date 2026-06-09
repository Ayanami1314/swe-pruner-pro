"""Fallback parser for models that emit tool calls as XML text in the response
body instead of the structured `tool_calls` field.

Context: `mimo-v2-flash` on SGLang returns

    msg.content = None
    msg.tool_calls = None
    msg.reasoning_content = "I'll list files.<tool_call>\\n<function=bash>\\n"
                            "<parameter=command>ls</parameter>\\n</function>\\n</tool_call>"

Our agent loop checks `msg.tool_calls` and gives up, so the agent never executes
anything. `parse_reasoning_tool_calls` converts the XML into an OpenAI-shaped
`tool_calls` list that the loop can consume.

The helper is opt-in: `run_agent_loop` takes a `tool_call_fallback_parser`
argument, which each benchmark wires to `maybe_mimo_parser(model)` — other
models are unaffected.
"""

from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAM_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)


def parse_reasoning_tool_calls(text: str | None, id_prefix: str = "mimo_") -> list[dict]:
    """Parse `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`
    blocks out of `text` and return them in OpenAI `tool_calls` shape.

    Returns `[]` if no complete block is found. Parameters without `=KEY`
    (e.g. bare `<parameter>`) cause the whole call to be skipped — OpenAI tools
    are keyword-only so there's no safe positional fallback. Literal
    `</parameter>` inside a value is a known gap; in practice mimo encodes
    values cleanly enough that this hasn't tripped.
    """
    if not text:
        return []
    out: list[dict] = []
    for i, m in enumerate(_TOOL_CALL_RE.finditer(text)):
        name = m.group(1)
        body = m.group(2)
        kwargs: dict[str, str] = {}
        ok = True
        for pm in _PARAM_RE.finditer(body):
            key = pm.group(1)
            if not key:
                ok = False
                break
            kwargs[key] = pm.group(2)
        if not ok:
            continue
        out.append({
            "id": f"{id_prefix}{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(kwargs, ensure_ascii=False)},
        })
    return out


def maybe_mimo_parser(model: str | None):
    """Return `parse_reasoning_tool_calls` when the model likely needs it, else None."""
    if not model:
        return None
    m = model.lower().lstrip("/")
    name = m.split("/")[-1]
    if name.startswith("mimo"):
        return parse_reasoning_tool_calls
    return None


# ── Inline tests ─────────────────────────────────────────────────────────

def _run_tests() -> None:
    # 1. Single tool call with one param
    t1 = ("I will list files.<tool_call>\n<function=bash>\n"
          "<parameter=command>ls -la</parameter>\n</function>\n</tool_call>")
    r1 = parse_reasoning_tool_calls(t1)
    assert len(r1) == 1, r1
    assert r1[0]["function"]["name"] == "bash"
    assert json.loads(r1[0]["function"]["arguments"]) == {"command": "ls -la"}
    assert r1[0]["id"] == "mimo_0"

    # 2. Multiple tool calls in one response
    t2 = ("<tool_call><function=bash><parameter=command>pwd</parameter></function></tool_call>"
          "then<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>")
    r2 = parse_reasoning_tool_calls(t2)
    assert len(r2) == 2, r2
    assert [tc["id"] for tc in r2] == ["mimo_0", "mimo_1"]
    assert json.loads(r2[1]["function"]["arguments"]) == {"command": "ls"}

    # 3. Multi-line parameter value (code snippet)
    t3 = ("<tool_call>\n<function=bash>\n<parameter=command>"
          "python3 -c '\nimport json\nprint(json.dumps({1:2}))\n'"
          "</parameter>\n</function>\n</tool_call>")
    r3 = parse_reasoning_tool_calls(t3)
    assert len(r3) == 1
    args = json.loads(r3[0]["function"]["arguments"])
    assert "import json" in args["command"]
    assert "print" in args["command"]

    # 4. Multiple params in one call
    t4 = ("<tool_call><function=bash>"
          "<parameter=command>cat x</parameter>"
          "<parameter=output_threshold>0.2</parameter>"
          "</function></tool_call>")
    r4 = parse_reasoning_tool_calls(t4)
    assert len(r4) == 1
    args4 = json.loads(r4[0]["function"]["arguments"])
    assert args4 == {"command": "cat x", "output_threshold": "0.2"}

    # 5. Missing closing </tool_call>  → no match → empty list
    t5 = "<tool_call><function=bash><parameter=command>ls</parameter></function>"
    assert parse_reasoning_tool_calls(t5) == []

    # 6. None / empty string
    assert parse_reasoning_tool_calls(None) == []
    assert parse_reasoning_tool_calls("") == []
    assert parse_reasoning_tool_calls("no xml here") == []

    # 7. Chinese text inside parameter
    t7 = ("<tool_call><function=bash>"
          "<parameter=command>echo '你好'</parameter>"
          "</function></tool_call>")
    r7 = parse_reasoning_tool_calls(t7)
    assert json.loads(r7[0]["function"]["arguments"]) == {"command": "echo '你好'"}

    # 8. maybe_mimo_parser model matching
    assert maybe_mimo_parser("mimo-v2-flash") is parse_reasoning_tool_calls
    assert maybe_mimo_parser("MIMO-PRO") is parse_reasoning_tool_calls
    assert maybe_mimo_parser("/some/path/mimo-xyz") is parse_reasoning_tool_calls
    assert maybe_mimo_parser("Qwen3-Coder-30B") is None
    assert maybe_mimo_parser("/mnt/user-ssd/.../Qwen3-Coder-30B-A3B-Instruct") is None
    assert maybe_mimo_parser(None) is None
    assert maybe_mimo_parser("") is None

    print("mimo: all tests passed")


if __name__ == "__main__":
    _run_tests()
