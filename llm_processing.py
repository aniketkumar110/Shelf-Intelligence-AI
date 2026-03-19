"""
llm_processing.py
Production-grade LLM orchestration using AutoGen + Azure OpenAI.

Key improvements:
- No hard-coded secrets (loads from env / key.py fallback)
- Builds agents via a single factory (create_llm_orchestrator)
- Robust JSON parsing (handles fenced JSON, stray text)
- Clear API: orchestrator.run(user_query, image_paths) -> dict
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CONFIG (env-first, optional key.py fallback)
# ---------------------------------------------------------------------
def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val or ""


def load_settings() -> Dict[str, str]:
    """
    Loads Azure OpenAI settings from environment variables.
    Falls back to key.py if present (for local dev).
    """
    settings = {
        "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE", "0"),
        "AGENT_WORK_DIR": os.getenv("AGENT_WORK_DIR", "Routing_File"),
    }

    # Optional fallback to key.py (do not require it in production)
    if not settings["AZURE_OPENAI_DEPLOYMENT_NAME"] or not settings["AZURE_OPENAI_ENDPOINT"]:
        try:
            # NOTE: keep names consistent with your existing key.py
            from key import (  # type: ignore
                AZURE_OPENAI_DEPLOYMENT_NAME,
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_API_VERSION,
            )
            settings["AZURE_OPENAI_DEPLOYMENT_NAME"] = settings["AZURE_OPENAI_DEPLOYMENT_NAME"] or AZURE_OPENAI_DEPLOYMENT_NAME
            settings["AZURE_OPENAI_ENDPOINT"] = settings["AZURE_OPENAI_ENDPOINT"] or AZURE_OPENAI_ENDPOINT
            settings["AZURE_OPENAI_API_KEY"] = settings["AZURE_OPENAI_API_KEY"] or AZURE_OPENAI_API_KEY
            settings["AZURE_OPENAI_API_VERSION"] = settings["AZURE_OPENAI_API_VERSION"] or AZURE_OPENAI_API_VERSION
        except Exception:
            pass

    # Production requires these to be present
    if not settings["AZURE_OPENAI_DEPLOYMENT_NAME"]:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME is missing (env or key.py).")
    if not settings["AZURE_OPENAI_ENDPOINT"]:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is missing (env or key.py).")
    if not settings["AZURE_OPENAI_API_KEY"]:
        raise RuntimeError("AZURE_OPENAI_API_KEY is missing (env or key.py).")

    return settings


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def encode_image_b64(image_path: str) -> str:
    """Base64-encodes an image file for Azure OpenAI vision input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def json_safe_loads(text: str) -> Dict[str, Any]:
    """
    Parses JSON from an LLM response robustly:
    - supports fenced JSON blocks
    - supports extra text around JSON (extracts first {...} block)
    """
    text = text.strip()

    # Case 1: fenced JSON
    m = _JSON_FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))

    # Case 2: pure JSON
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # Case 3: extract first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Could not parse JSON from LLM response.")


def prepare_message_single(image_paths: List[str], user_query: str) -> Dict[str, Any]:
    """
    Creates the single user message containing:
    - User query text
    - Multiple row images (cropped / annotated) in order
    """
    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": f"User Query:\n{user_query}\nShelf rows follow (Row 1, Row 2, ... in order):"
        }
    ]

    for img_path in image_paths:
        encoded_img = encode_image_b64(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"},
            }
        )

    return {"role": "user", "content": content}


# ---------------------------------------------------------------------
# PROMPTS (kept as constants for maintainability)
# ---------------------------------------------------------------------
ROUTING_PROMPT = r"""
You are a Routing Agent.

Decide whether the user query requires numeric counting
or only visual / descriptive understanding.

Choose "Count_Agent" ONLY if the user explicitly asks for numbers,
totals or quantities, such as:
- how many
- count
- total number
- number of items

Otherwise choose "Generic_Agent".

Return STRICT JSON only:

{
  "next_agent": "Count_Agent" | "Generic_Agent",
  "reason": "<short explanation>"
}

No extra text.
""".strip()

COUNT_PROMPT = r"""
You are a Deterministic Retail Shelf Counting Agent.
Behave like a rule-based visual auditor, NOT a guesser.

IMPORTANT:
- MULTIPLE shelf-row images in ONE message (each corresponds to one row)

Rules:
1) Process all row images before responding.
2) Count ONLY if 50%+ of front face is visible.
3) No guessing / no inference outside the frame.
4) If user requests total items: return counts PER BRAND and compute totals internally.
5) If row is 60-80%+ truncated/blurry: counts must be {} and explain why.

Output STRICT JSON ONLY:

{
  "row_results": [
    {
      "row": <row_number>,
      "counts": { "<brand_or_product_name>": <integer> },
      "reasoning": "<brief explanation>"
    }
  ],
  "next_agent": "Final_Answer_Agent"
}
""".strip()

GENERIC_PROMPT = r"""
You are a Generic Shelf Understanding Agent.

CRITICAL:
- The user uploads ONLY ONE shelf image.
- Any multiple images you receive are CROPPED ROWS derived internally.
- DO NOT imply the user uploaded multiple images.

Use row numbers (Row 1, Row 2, ...) for placement.

Return STRICT JSON ONLY:

{
  "Answer": "<clear natural language answer>",
  "reasoning": "<brief visual explanation referencing row numbers>",
  "next_agent": "Final_Answer_Agent"
}
""".strip()

FINAL_PROMPT = r"""
You are the Final Answer Agent.

INPUT:
- Original user query
- Aggregated results from previous agent

TASK:
- If user asks total in whole image, merge row counts.
- If user asks per row, format per row.
- If comparisons, rank/compare.

Return STRICT JSON only:

{
  "user_query":"<original query>",
  "final_answer":"<clear natural language answer>",
  "reasoning":"<short explanation>"
}
""".strip()


# ---------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------
@dataclass
class LLMOrchestrator:
    user_proxy: UserProxyAgent
    manager: GroupChatManager

    def run(self, user_query: str, image_paths: List[str], clear_history: bool = True) -> Dict[str, Any]:
        """
        Runs the agentic chain and returns the parsed final JSON.
        """
        message = prepare_message_single(image_paths, user_query)

        self.user_proxy.initiate_chat(
            self.manager,
            message=message,
            clear_history=clear_history,
        )

        final_raw = self.manager.groupchat.messages[-1]["content"]
        return json_safe_loads(final_raw)


def create_llm_orchestrator() -> LLMOrchestrator:
    """
    Factory to build all agents and return a single orchestrator object.
    Call this once and cache it (Streamlit cache_resource).
    """
    s = load_settings()

    llm_config = {
        "config_list": [
            {
                "model": s["AZURE_OPENAI_DEPLOYMENT_NAME"],
                "api_type": "azure",
                "base_url": s["AZURE_OPENAI_ENDPOINT"],
                "api_key": s["AZURE_OPENAI_API_KEY"],
                "api_version": s["AZURE_OPENAI_API_VERSION"],
            }
        ],
        "temperature": float(s["LLM_TEMPERATURE"]),
    }

    user_proxy = UserProxyAgent(
        name="user_proxy",
        system_message=(
            "You are the User Proxy.\n"
            "You will send shelf-row images along with the user query.\n"
            "Do not reason or modify content."
        ),
        code_execution_config={"work_dir": s["AGENT_WORK_DIR"], "use_docker": False},
        max_consecutive_auto_reply=5,
        llm_config=llm_config,
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: False,
    )

    routing_agent = AssistantAgent(
        name="Routing_Agent",
        system_message=ROUTING_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    count_agent = AssistantAgent(
        name="Count_Agent",
        system_message=COUNT_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    generic_agent = AssistantAgent(
        name="Generic_Agent",
        system_message=GENERIC_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    final_agent = AssistantAgent(
        name="Final_Answer_Agent",
        system_message=FINAL_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    def state_transition(last_speaker, groupchat):
        """
        Robust state transitions driven by the JSON output of each agent.
        """
        if last_speaker is user_proxy:
            return routing_agent

        # If something went wrong, stop the chat rather than looping forever
        if not groupchat.messages:
            return None

        last_msg = groupchat.messages[-1].get("content", "")
        try:
            parsed = json_safe_loads(last_msg)
            next_agent_name = parsed.get("next_agent")
        except Exception:
            logger.exception("Failed to parse agent JSON. Stopping agent flow.")
            return None

        if next_agent_name == "Count_Agent":
            return count_agent
        if next_agent_name == "Generic_Agent":
            return generic_agent
        if next_agent_name == "Final_Answer_Agent":
            return final_agent

        return None

    groupchat = GroupChat(
        agents=[user_proxy, routing_agent, count_agent, generic_agent, final_agent],
        messages=[],
        max_round=10,  # production: keep tight to prevent runaway costs
        speaker_selection_method=state_transition,
    )

    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    return LLMOrchestrator(user_proxy=user_proxy, manager=manager)
