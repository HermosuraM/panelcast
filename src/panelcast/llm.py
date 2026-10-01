"""Claude chat-model factory (LangChain). Refusal fallbacks are on by default; sampling params are not
set because current Claude models reject them - depth is controlled with `effort` instead."""

from __future__ import annotations

import os

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _load_databricks_secret() -> None:
    """On Databricks, read the key from secret scope `panelcast` (key `anthropic_api_key`) if it exists."""
    if "DATABRICKS_RUNTIME_VERSION" not in os.environ or os.environ.get("ANTHROPIC_API_KEY"):
        return
    try:
        from databricks.sdk.runtime import dbutils

        os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("panelcast", "anthropic_api_key")
    except Exception:  # noqa: BLE001 - no scope / no key: LLM steps fall back to deterministic logic
        pass


def llm_available() -> bool:
    _load_databricks_secret()
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def llm_enabled(flag: str | bool) -> bool:
    if isinstance(flag, bool):
        return flag
    return str(flag).lower() == "true" or (str(flag).lower() == "auto" and llm_available())


def chat_model(model: str, effort: str = "medium", max_tokens: int = 16000):
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        effort=effort,
        betas=[FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},  # a policy decline is retried server-side on another model
        max_retries=3,
    )


def was_refused(message) -> bool:
    meta = getattr(message, "response_metadata", {}) or {}
    return meta.get("stop_reason") == "refusal"
