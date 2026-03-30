from __future__ import annotations

import base64
from typing import Any

import numpy as np

from planner_eval.core.engine import OpenAIChatEngine
from planner_eval.core.llm_trace import record_llm_input


class LMMAgent:
    """Thin message manager around an OpenAI-compatible multimodal chat engine."""

    def __init__(self, engine_params: dict[str, Any] | None = None, system_prompt=None, engine=None):
        if engine is None:
            if engine_params is None:
                raise ValueError("engine_params must be provided")
            engine_type = engine_params.get("engine_type", "openai")
            if engine_type not in {"openai", "openai_compatible"}:
                raise ValueError(f"Unsupported engine_type: {engine_type}")
            self.engine = OpenAIChatEngine(**engine_params)
        else:
            self.engine = engine

        self.messages = []
        self.add_system_prompt(system_prompt or "You are a helpful assistant.")

    def encode_image(self, image_content):
        if isinstance(image_content, str):
            with open(image_content, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        return base64.b64encode(image_content).decode("utf-8")

    def reset(self):
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def add_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        if self.messages:
            self.messages[0] = {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        else:
            self.messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )

    def remove_message_at(self, index):
        if index < len(self.messages):
            self.messages.pop(index)

    def replace_message_at(
        self,
        index,
        text_content,
        image_content=None,
        image_detail="high",
    ):
        if index >= len(self.messages):
            return
        self.messages[index] = {
            "role": self.messages[index]["role"],
            "content": [{"type": "text", "text": text_content}],
        }
        if image_content:
            base64_image = self.encode_image(image_content)
            self.messages[index]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": image_detail,
                    },
                }
            )

    def _infer_role(self, explicit_role):
        if explicit_role == "user":
            return explicit_role
        if self.messages[-1]["role"] == "system":
            return "user"
        if self.messages[-1]["role"] == "user":
            return "assistant"
        return "user"

    def add_message(
        self,
        text_content,
        image_content=None,
        role=None,
        image_detail="high",
        put_text_last=False,
    ):
        role = self._infer_role(role)
        message = {
            "role": role,
            "content": [{"type": "text", "text": text_content}],
        }

        if isinstance(image_content, np.ndarray) or image_content:
            images = image_content if isinstance(image_content, list) else [image_content]
            for image in images:
                base64_image = self.encode_image(image)
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": image_detail,
                        },
                    }
                )

        if put_text_last:
            text_part = message["content"].pop(0)
            message["content"].append(text_part)

        self.messages.append(message)

    def get_response(
        self,
        user_message=None,
        messages=None,
        temperature=0.0,
        max_new_tokens=None,
        use_thinking=False,
        **kwargs,
    ):
        if messages is None:
            messages = self.messages
        if user_message:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": user_message}]}
            )

        try:
            record_llm_input(provider_tag=type(self.engine).__name__, messages=messages)
        except Exception:
            pass

        if use_thinking:
            return self.engine.generate_with_thinking(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        return self.engine.generate(
            messages,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
