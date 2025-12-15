"""
Defense utilities
"""

# pylint: disable=W0102

from collections.abc import Sequence
from typing import Any

from typing_extensions import Literal

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.pi_detector import DetectorTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger, TraceLogger
from agentdojo.types import (
    ChatMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)


def push_contextarg(logger: TraceLogger, key: str, value: Any):
    """
    Push an item to a context argument list
    """

    if key in logger.context:
        if isinstance(logger.context[key], list):
            logger.context[key].append(value)
        else:
            raise ValueError(
                f"Expected TraceLogger context key {key} to be a list, got: {type(logger.context[key])}"
            )
    else:
        logger.context[key] = [value]


class EnhancedPromptInjectionDetector(BasePipelineElement):
    """
    An enhanced version of the `PromptInjectionDetector` that automatically retrieves the trace
    logger and supports passing the full conversation history to the detector.
    """

    def __init__(self, mode: Literal["message", "full_conversation"] = "message"):
        super().__init__()
        self.mode = mode

    def transform(
        self, tool_output: list[MessageContentBlock]
    ) -> list[MessageContentBlock]:
        """
        Transforms the tool output by replacing text blocks with a placeholder indicating that data
        has been omitted due to prompt injection detection.
        """

        return [
            (
                text_content_block_from_string(
                    "<Data omitted because a prompt injection was detected>"
                )
                if block["type"] == "text"
                else block
            )
            for block in tool_output
        ]

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        """
        Pipeline query function
        """

        # Check for messages
        if len(messages) == 0:
            return query, runtime, env, messages, extra_args

        # Skip if last message is not from a tool
        if messages[-1]["role"] != "tool":
            return query, runtime, env, messages, extra_args

        # Get the messages to check
        messages_to_check = [
            DetectorTask(
                len(messages) - 1,
                messages[-1],
                get_text_content_as_str(messages[-1]["content"]) or "",
            )
        ]
        n_tool_results = 1
        for i, message in reversed(list(enumerate(messages[:-1]))):
            if message["role"] != "tool":
                break
            messages_to_check.append(
                DetectorTask(
                    i, message, get_text_content_as_str(message["content"]) or ""
                )
            )
            n_tool_results += 1

        # Get the logger
        logger = Logger().get()

        if not isinstance(logger, TraceLogger):
            raise ValueError(f"Expected TraceLogger, got: {type(logger)}")

        # Detect injections
        processed_messages = list(messages[:-n_tool_results])
        for task in messages_to_check:
            previous_messages = [message for message in messages[: task.index + 1]]
            detection_result = self.detect(task.text, previous_messages, logger)
            score = (
                (1.0 if detection_result else 0.0)
                if not isinstance(detection_result, tuple)
                else detection_result[1]
            )
            is_injection = (
                detection_result
                if not isinstance(detection_result, tuple)
                else detection_result[0]
            )

            if is_injection:
                raise AbortAgentError(
                    f"aborting execution because a prompt injection was detected (score: {score}) in message: {task.text}",
                    list(messages),
                    env,
                )

            if is_injection:
                task.message["content"] = self.transform(task.message["content"] or [])
            processed_messages.append(task.message)

        return query, runtime, env, messages, extra_args

    def detect(
        self, tool_output: str, messages: Sequence[ChatMessage], logger: TraceLogger
    ) -> bool | tuple[bool, float]:
        """
        Returns True if a prompt injection is detected, False otherwise.

        Raises:
            AbortAgentError: if an injection is detected and if `self.raise_on_injection` is `True`.

        Returns:
            If the detector is a probabilistic one, it can return a tuple with the first element being a boolean
                indicating whether a prompt injection was detected, and the second element being a float representing
                the confidence score. If the detector is not probabilistic, it should return a boolean.
        """
        raise NotImplementedError(
            "to use EnhancedPromptInjectionDetector, use a subclass with a detect method"
        )
