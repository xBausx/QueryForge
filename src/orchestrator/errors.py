# src/orchestrator/errors.py
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """
    Authoritative error codes per Hard I/O Contract.
    """
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    MISSING_PARAMETER     = "MISSING_PARAMETER"
    AMBIGUOUS_REQUEST     = "AMBIGUOUS_REQUEST"
    SCHEMA_MISSING        = "SCHEMA_MISSING"
    SCHEMA_MISMATCH       = "SCHEMA_MISMATCH"
    FORBIDDEN_FIELD       = "FORBIDDEN_FIELD"


class OrchestratorError(Exception):
    """
    Structured error used across the pipeline. When surfaced to the caller,
    you must format it as: {"error": "[CODE] <reason>"}.

    Internal services may also attach a lightweight trace/meta dict; this
    MUST NOT leak into the top-level "error" string, but can be returned
    alongside it under a separate key if desired (e.g., "trace").
    """
    def __init__(self, code: ErrorCode | str, message: str, *, meta: Optional[Dict[str, Any]] = None):
        if isinstance(code, str):
            # Allow raising with a string like "[MISSING_PARAMETER]" or "MISSING_PARAMETER"
            code = code.strip().strip("[]")
            try:
                code = ErrorCode[code]
            except KeyError:
                # Fallback to UNSUPPORTED_OPERATION for unknown codes
                code = ErrorCode.UNSUPPORTED_OPERATION

        self.code: ErrorCode = code
        self.message: str = message
        self.meta: Dict[str, Any] = meta or {}

        super().__init__(f"[{self.code.value}] {self.message}")

    def to_public_json(self) -> Dict[str, str]:
        """
        Conform to the strict external shape.
        """
        return {"error": f"[{self.code.value}] {self.message}"}

    def with_trace(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Utility: produce a combined payload with the required error string
        plus a trace object for debugging/UX (still deterministic & lightweight).
        """
        return {**self.to_public_json(), "trace": trace}


# ---------------------------
# Convenience constructors
# ---------------------------

def error_json(code: ErrorCode | str, message: str) -> Dict[str, str]:
    """
    Create the exact wire-format error object:
    {"error": "[CODE] message"}
    """
    if isinstance(code, str):
        code = code.strip().strip("[]")
        try:
            code_enum = ErrorCode[code]
        except KeyError:
            code_enum = ErrorCode.UNSUPPORTED_OPERATION
    else:
        code_enum = code
    return {"error": f"[{code_enum.value}] {message}"}


def raise_missing_parameter(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.MISSING_PARAMETER, message, meta=meta)


def raise_ambiguous_request(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.AMBIGUOUS_REQUEST, message, meta=meta)


def raise_schema_missing(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.SCHEMA_MISSING, message, meta=meta)


def raise_schema_mismatch(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.SCHEMA_MISMATCH, message, meta=meta)


def raise_forbidden_field(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.FORBIDDEN_FIELD, message, meta=meta)


def raise_unsupported_operation(message: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
    raise OrchestratorError(ErrorCode.UNSUPPORTED_OPERATION, message, meta=meta)
