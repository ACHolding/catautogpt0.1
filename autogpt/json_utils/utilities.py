"""Utilities for the json_fixes package."""
import ast
import json
import os.path
import re
from typing import Any, Literal

from jsonschema import Draft7Validator

from autogpt.config import Config
from autogpt.logs import logger

LLM_DEFAULT_RESPONSE_FORMAT = "llm_response_format_1"

_FENCE_RE = re.compile(
    r"^```(?:json|javascript|js|python|py)?\s*\n?(.*?)\n?```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """Remove markdown fences, including ```json language tags."""
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    match = _FENCE_RE.match(cleaned)
    if match:
        return match.group(1).strip()
    # Fallback: drop first/last fence lines even if regex missed.
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _loads_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON/Python-dict object string into a dict."""
    cleaned = text.strip()
    if not cleaned:
        return None

    # Local LLMs (CatSeek) emit real JSON; OpenAI historically emitted Python dicts.
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    try:
        value = ast.literal_eval(cleaned)
        if isinstance(value, dict):
            return value
    except (SyntaxError, ValueError, MemoryError):
        pass

    return None


def extract_dict_from_response(response_content: str) -> dict[str, Any]:
    """Extract a dict from an LLM reply (JSON, Python dict, or fenced block)."""
    if response_content is None:
        return {}

    text = _strip_code_fence(str(response_content))
    parsed = _loads_object(text)
    if parsed is not None:
        return parsed

    # Model often adds a preface; take the outermost {...} span.
    match = _OBJECT_RE.search(text)
    if match:
        parsed = _loads_object(match.group(0))
        if parsed is not None:
            return parsed

    logger.info(
        "Error parsing JSON response with literal_eval "
        f"invalid syntax near: {text[:120]!r}"
    )
    logger.debug(f"Invalid JSON received in response: {response_content}")
    return {}


def llm_response_schema(
    config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> dict[str, Any]:
    filename = os.path.join(os.path.dirname(__file__), f"{schema_name}.json")
    with open(filename, "r") as f:
        try:
            json_schema = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON schema: {e}")
    if config.openai_functions:
        del json_schema["properties"]["command"]
        json_schema["required"].remove("command")
    return json_schema


def validate_dict(
    object: object, config: Config, schema_name: str = LLM_DEFAULT_RESPONSE_FORMAT
) -> tuple[Literal[True], None] | tuple[Literal[False], list]:
    """
    :type schema_name: object
    :param schema_name: str
    :type json_object: object

    Returns:
        bool: Whether the json_object is valid or not
        list: Errors found in the json_object, or None if the object is valid
    """
    schema = llm_response_schema(config, schema_name)
    validator = Draft7Validator(schema)

    if errors := sorted(validator.iter_errors(object), key=lambda e: e.path):
        for error in errors:
            logger.debug(f"JSON Validation Error: {error}")

        if config.debug_mode:
            logger.error(json.dumps(object, indent=4))
            logger.error("The following issues were found:")

            for error in errors:
                logger.error(f"Error: {error.message}")
        return False, errors

    logger.debug("The JSON object is valid.")

    return True, None
