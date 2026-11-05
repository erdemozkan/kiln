"""
Adds tool-calling and document cases to chat-templates.json, rendered by Python jinja2 the way
transformers' apply_chat_template does (tools=None and documents=None when not given). Also checks
that every case already in the file renders the same with those None values passed.

    python3 add-tool-cases.py chat-templates.json   # needs jinja2 >= 3.1, as transformers does
"""

import json
import sys

from jinja2 import nodes
from jinja2.ext import Extension, loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the local time in a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
TOOLS = [WEATHER, TIME]
DOCUMENTS = [
    {"doc_id": 1, "title": "Opening hours", "text": "The museum opens at 9am and closes at 5pm."},
    {"doc_id": 2, "title": "Tickets", "text": "Tickets cost 12 euros; under-18s go free."},
]


def call(id, name, args):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


CONVERSATIONS = {
    "tool-call": [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [call("call12345", "get_weather", {"city": "Paris"})]},
        {"role": "tool", "tool_call_id": "call12345", "name": "get_weather", "content": '{"temperature": 18, "unit": "celsius"}'},
    ],
    "tool-parallel": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather and time in Rome?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [call("callA1234", "get_weather", {"city": "Rome", "unit": "celsius"}), call("callB5678", "get_time", {"city": "Rome"})],
        },
        {"role": "tool", "tool_call_id": "callA1234", "name": "get_weather", "content": '{"temperature": 24}'},
        {"role": "tool", "tool_call_id": "callB5678", "name": "get_time", "content": "14:05"},
        {"role": "assistant", "content": "24 °C and 14:05."},
        {"role": "user", "content": "Thanks!"},
    ],
    "museum": [{"role": "user", "content": "When does the museum open, and what does it cost?"}],
}

# Templates that read `tools` (and, for granite, `documents`), with the cases each gets.
CASES = {
    "smollm3": [("single", {"tools": TOOLS}), ("tool-call", {"tools": TOOLS}), ("tool-parallel", {"tools": TOOLS, "enable_thinking": False})],
    "qwen2.5": [("single", {"tools": TOOLS}), ("system", {"tools": TOOLS}), ("tool-call", {"tools": TOOLS}), ("tool-parallel", {"tools": TOOLS})],
    "qwen3": [("single", {"tools": TOOLS}), ("tool-call", {"tools": TOOLS}), ("tool-parallel", {"tools": TOOLS, "enable_thinking": False})],
    "mistral-v0.3": [("single", {"tools": TOOLS}), ("tool-call", {"tools": TOOLS}), ("tool-parallel", {"tools": TOOLS})],
    "granite3.3": [
        ("single", {"tools": TOOLS}),
        ("tool-call", {"tools": TOOLS}),
        ("tool-parallel", {"tools": TOOLS}),
        ("museum", {"documents": DOCUMENTS}),
        ("museum", {"tools": TOOLS, "documents": DOCUMENTS}),
    ],
}


class TemplateError(Exception):
    pass


def raise_exception(msg):
    raise TemplateError(msg)


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Generation(Extension):
    """`{% generation %}…{% endgeneration %}` as transformers renders it when not tracking: the body."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols, Generation])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception


# Pinned like the existing cases (and the test's globals): one date whatever the format.
env.globals["strftime_now"] = lambda f: "26 Jul 2024"


def render(tpl, convs, conv, vars):
    full = {"tools": None, "documents": None, **vars, "messages": convs[conv]}
    try:
        return {"text": tpl.render(**full)}
    except TemplateError as e:
        return {"error": str(e)}


def main(path):
    fx = json.load(open(path))
    fx["conversations"].update(CONVERSATIONS)
    for t in fx["templates"]:
        tpl = env.from_string(t["template"])
        for c in t["cases"]:
            got = render(tpl, fx["conversations"], c["messages"], c["vars"])
            want = {k: c[k] for k in ("text", "error") if k in c}
            if got != want:
                sys.exit(f"{t['name']} {c['messages']}: renders differently with tools=None")
        if t["name"] not in CASES:
            continue
        base = next(c["vars"] for c in t["cases"])
        base = {k: base[k] for k in ("add_generation_prompt", "bos_token", "eos_token")}
        # Rerunning replaces the cases this script added earlier.
        t["cases"] = [c for c in t["cases"] if "tools" not in c["vars"] and "documents" not in c["vars"]]
        for conv, vars in CASES[t["name"]]:
            v = {**base, **vars}
            t["cases"].append({"messages": conv, "vars": v, **render(tpl, fx["conversations"], conv, v)})
            print(t["name"], conv, "error" if "error" in t["cases"][-1] else "ok")
    with open(path, "w") as f:
        json.dump(fx, f, ensure_ascii=False, indent=1)


main(sys.argv[1])
