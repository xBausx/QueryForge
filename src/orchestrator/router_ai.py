# src/orchestrator/router_ai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import os
import json

from .models import LogicalQueryRequest, Direction, OrderBy  # for validation
from .date_normalizer import infer_time_range

def _build_enums(config: Any) -> Dict[str, List[str]]:
    metrics = sorted(list((getattr(config, "metrics", {}) or {}).keys()))
    dimensions = sorted(list((getattr(config, "dimensions", {}) or {}).keys()))
    fields = sorted(list(set(dimensions) | set((getattr(config, "fields", {}) or {}).keys())))
    resolvers = sorted(list((getattr(config, "resolvers", {}) or {}).keys()))
    return {
        "metrics": metrics,
        "dimensions": dimensions,
        "fields": fields,
        "resolvers": resolvers,
    }

def _json_schema(enums: Dict[str, List[str]]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "metrics": {"type": "array", "items": {"enum": enums["metrics"]}, "minItems": 1},
            "dimensions": {"type": "array", "items": {"enum": enums["dimensions"]}},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"enum": enums["fields"]},
                        "op": {"enum": ["eq","neq","gt","lt","gte","lte","between","in","like"]},
                        "value": {}
                    },
                    "required": ["field","op","value"],
                    "additionalProperties": False
                }
            },
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_fk": {"enum": enums["resolvers"]},
                        "value": {"type": "string"},
                        "mode": {"enum": ["ilike_contains","ilike_prefix","exact"]},
                    },
                    "required": ["target_fk","value"],
                    "additionalProperties": False
                }
            },
            "order_by": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"enum": enums["metrics"] + enums["dimensions"]},
                        "direction": {"enum": ["asc","desc"]}
                    },
                    "required": ["field","direction"],
                    "additionalProperties": False
                }
            },
            "time_range": {
                "type": "object",
                "properties": {"start": {"type":"string"}, "end": {"type":"string"}},
                "required": ["start","end"],
                "additionalProperties": False
            },
            "limit": {"type":"integer"},
            "error": {"type":"string"},
            "clarify": {"type":"string"},
            "suggestions": {"type":"array","items":{"type":"string"}},
        },
        "required": [],
        "additionalProperties": False
    }

_SYSTEM_PROMPT = (
    "You are a STRICT router that converts a user's natural-language analytics question into a JSON object "
    "for a SELECT-only query builder. Return ONLY JSON that conforms to the provided schema. Never output SQL. "
    "Use ONLY the allowed enums for metrics, dimensions, fields, and resolvers. "
    "If required info is missing, return an object with 'error':'MISSING_PARAMETER' and a short 'clarify'. "
    "If a token maps to multiple dimensions on the same table, return 'error':'AMBIGUOUS_REQUEST' with 'suggestions'. "
    "Do NOT invent fields or metrics."
)

_FEW_SHOTS = [
    {
        "q": "top 5 dealers by installs in texas last 30 days",
        "a": {
            "metrics": ["licenses.count"],
            "dimensions": ["licenses.dealer_id"],
            "filters": [{"field":"licenses.state","op":"eq","value":"TX"}],
            "order_by": [{"field":"licenses.count","direction":"desc"}],
            "limit": 5
        }
    },
    {
        "q": "all hosts per state for dealer orion",
        "a": {
            "metrics":["hosts.count"],
            "dimensions":["hosts.state"],
            "resolutions":[{"target_fk":"hosts.dealer_id","value":"orion"}],
            "limit": 100
        }
    }
]

def _build_messages(nl_text: str, tz: str, schema: Dict[str, Any]) -> list[Dict[str, str]]:
    shots = "\n---\n".join([f"Q: {s['q']}\nA: {json.dumps(s['a'])}" for s in _FEW_SHOTS])
    user = f"Question: {nl_text}\nTimezone: {tz}\nReturn JSON only."
    return [
        {"role": "system", "content": _SYSTEM_PROMPT + "\nJSON schema will be provided as a tool/function."},
        {"role": "user", "content": shots},
        {"role": "user", "content": user},
    ]

def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict validation using our Pydantic LogicalQueryRequest. We allow partials:
    - If time_range missing but dates mentioned, the orchestrator/date_normalizer will fill.
    - If limit missing, downstream will inject default 100.
    """
    # Coerce to LogicalQueryRequest; this raises on any off-allowlist / wrong shape
    lqr = LogicalQueryRequest(**{
        "metrics": payload.get("metrics") or [],
        "dimensions": payload.get("dimensions") or [],
        "filters": payload.get("filters") or [],
        "order_by": payload.get("order_by") or [],
        "limit": payload.get("limit"),
        "time_range": payload.get("time_range"),
        "resolutions": payload.get("resolutions") or [],
    })
    # return as dict for orchestrator to pass to AST
    return lqr.model_dump()

def translate(nl_text: str, config: Any, *, tz: str = "Asia/Manila", provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Translate NL -> LogicalQueryRequest JSON using an LLM (JSON-only), with strict schema/enums.
    Providers:
      - openai (function-calling)
      - gemini (application/json constrained)
    Auto-picks provider from ROUTER_PROVIDER env if not specified.
    Never returns SQL. Never blocks pipeline: emits a clear error on failure.
    """
    enums = _build_enums(config)
    schema = _json_schema(enums)
    messages = _build_messages(nl_text, tz, schema)

    prov = (provider or os.getenv("ROUTER_PROVIDER") or "openai").lower().strip()

    # ---------- OpenAI path ----------
    if prov == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "error": "[MISSING_PARAMETER] AI router not available (OPENAI_API_KEY not set).",
                "trace": {"ai": "no_api_key", "provider": "openai"},
            }
        try:
            import openai  # type: ignore
        except Exception:
            return {
                "error": "[MISSING_PARAMETER] AI router not available (openai sdk not installed).",
                "trace": {"ai": "unavailable", "provider": "openai"},
            }

        try:
            client = openai.OpenAI() if hasattr(openai, "OpenAI") else None
            if client:
                resp = client.chat.completions.create(
                    model=os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini"),
                    temperature=0,
                    messages=messages,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "produce_logical_query",
                            "description": "Return a LogicalQueryRequest JSON for the query builder.",
                            "parameters": schema,
                        },
                    }],
                    tool_choice={"type": "function", "function": {"name": "produce_logical_query"}},
                )
                choice = resp.choices[0]
                tool_call = choice.message.tool_calls[0]
                args = json.loads(tool_call.function.arguments or "{}")
            else:
                # legacy SDK
                resp = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini"),
                    temperature=0,
                    messages=messages,
                    functions=[{
                        "name": "produce_logical_query",
                        "description": "Return a LogicalQueryRequest JSON for the query builder.",
                        "parameters": schema,
                    }],
                    function_call={"name": "produce_logical_query"},
                )
                args = json.loads(resp["choices"][0]["message"]["function_call"]["arguments"] or "{}")

            if args.get("error"):
                return {
                    "error": f"[{args.get('error')}] {args.get('clarify','')}".strip(),
                    "trace": {"ai": "returned-error", "provider": "openai", "suggestions": args.get("suggestions")},
                }

            lqr_dict = _validate_payload(args)
            return {"lqr": lqr_dict, "trace": {"ai": "ok", "router": "openai"}}

        except Exception as e:
            return {"error": f"[MISSING_PARAMETER] AI router failed: {e}", "trace": {"ai": "exception", "provider": "openai"}}

    # ---------- Gemini path ----------
    elif prov == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "error": "[MISSING_PARAMETER] AI router not available (GEMINI_API_KEY not set).",
                "trace": {"ai": "no_api_key", "provider": "gemini"},
            }
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
        except Exception:
            return {
                "error": "[MISSING_PARAMETER] AI router not available (google-generativeai not installed).",
                "trace": {"ai": "unavailable", "provider": "gemini"},
            }

        try:
            # Gemini doesn't have the same "tools" API; we embed the schema and demand JSON-only output.
            model = genai.GenerativeModel(os.getenv("GEMINI_ROUTER_MODEL", "gemini-1.5-pro"))
            payload = {
                "system": _SYSTEM_PROMPT + "\nReturn ONLY JSON per the schema below.\n" +
                        "SCHEMA:\n" + json.dumps(schema, ensure_ascii=False),
                "user": f"Question: {nl_text}\nTimezone: {tz}\nReturn JSON only."
            }
            resp = model.generate_content(
                [payload["system"], payload["user"]],
                generation_config={"temperature": 0, "response_mime_type": "application/json"},
            )
            text = getattr(resp, "text", None)
            args = json.loads(text or "{}")

            if args.get("error"):
                return {
                    "error": f"[{args.get('error')}] {args.get('clarify','')}".strip(),
                    "trace": {"ai": "returned-error", "provider": "gemini", "suggestions": args.get("suggestions")},
                }

            lqr_dict = _validate_payload(args)
            return {"lqr": lqr_dict, "trace": {"ai": "ok", "router": "gemini"}}

        except Exception as e:
            return {"error": f"[MISSING_PARAMETER] AI router failed: {e}", "trace": {"ai": "exception", "provider": "gemini"}}

    # ---------- Unknown provider ----------
    return {"error": "[MISSING_PARAMETER] AI router not available (unknown provider).", "trace": {"ai": "unknown"}}