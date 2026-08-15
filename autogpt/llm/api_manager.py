from __future__ import annotations

from typing import Any, List, Optional

from autogpt.llm.base import CompletionModelInfo
from autogpt.logs import logger
from autogpt.singleton import Singleton


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[dict[str, Any]]] = None

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None

    def update_cost(self, prompt_tokens, completion_tokens, model):
        """
        Update the total cost, prompt tokens, and completion tokens.
        Local BitNet inference is free; we still track token counts.
        """
        from autogpt.llm.providers.openai import OPEN_AI_MODELS

        model = model[:-3] if model.endswith("-v2") else model
        model_info = OPEN_AI_MODELS.get(model) or OPEN_AI_MODELS["bitnet-b1.58"]

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += prompt_tokens * model_info.prompt_token_cost / 1000
        if issubclass(type(model_info), CompletionModelInfo):
            self.total_cost += (
                completion_tokens * model_info.completion_token_cost / 1000
            )

        logger.debug(
            f"Tokens used — prompt: {prompt_tokens}, completion: {completion_tokens}; "
            f"running cost: ${self.total_cost:.3f}"
        )

    def set_total_budget(self, total_budget):
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        return self.total_completion_tokens

    def get_total_cost(self):
        return self.total_cost

    def get_total_budget(self):
        return self.total_budget

    def get_models(self, **_credentials) -> List[dict[str, Any]]:
        """Return locally available BitNet-compatible model ids."""
        if self.models is None:
            from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS

            self.models = [{"id": name} for name in OPEN_AI_CHAT_MODELS]
        return self.models
