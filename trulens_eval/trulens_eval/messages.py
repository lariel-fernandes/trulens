import json
import logging
from abc import ABC, abstractmethod
from typing import Iterable, Dict, List
from typing import Union, Type

from langchain.agents import AgentType
from langchain.schema import AgentFinish, AgentAction
from langchain.tools import BaseTool, Tool
from pydantic import Field

from trulens_eval.schema import MessageSource, MessageContentType
from trulens_eval.schema import Record, MessageInfo
from trulens_eval.util import SerialModel
from trulens_eval.util import jsonify, JSON

logger = logging.getLogger(__name__)


class MessagesExtractor(SerialModel, ABC):

    @abstractmethod
    def __call__(self, record: Record) -> Iterable[MessageInfo]:
        raise NotImplementedError()


class OpenAiFunctionsMessagesExtractor(MessagesExtractor):
    tools: Dict[str, dict]
    metadata: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata["agent_type"] = AgentType.OPENAI_FUNCTIONS

    @classmethod
    def from_tools(cls, tools: List[BaseTool], **kwargs):
        tools = {tool.name: jsonify_tool(tool) for tool in tools}
        return cls(tools=tools, **kwargs)

    def __call__(self, record: Record) -> Iterable[MessageInfo]:
        for idx, call in enumerate(record.calls):
            if call.method().name not in ["plan", "aplan"]:
                continue
            if idx == 0:
                yield MessageInfo(
                    source=MessageSource.USER,
                    label="Input",
                    content=coerce_dict_to_str(call.args["kwargs"]),
                    metadata_=self.metadata,
                    ts=call.perf.start_time,
                    call_idx=idx,
                )
            if steps := call.args["intermediate_steps"]:
                yield MessageInfo(
                    source=MessageSource.SYSTEM,
                    label="Observation",
                    content=steps[-1][1],
                    metadata_=self.metadata,
                    ts=call.perf.start_time,
                    call_idx=idx,
                )
            if isinstance(call.rets, AgentAction):
                yield MessageInfo(
                    source=MessageSource.ASSISTANT,
                    label=f"Action[{call.rets.tool}]",
                    content=coerce_dict_to_str(call.rets.tool_input, truncate_single_key=False),
                    metadata_={
                        **self.metadata,
                        "log": call.rets.log.strip(),
                        "tool_info": self.tools.get(call.rets.tool)
                    },
                    ts=call.perf.end_time,
                    call_idx=idx,
                    content_type=(
                        MessageContentType.JSON
                        if isinstance(call.rets.tool_input, (dict, list))
                        else MessageContentType.TEXT
                    )
                )
            elif isinstance(call.rets, AgentFinish):
                yield MessageInfo(
                    source=MessageSource.ASSISTANT,
                    label=call.rets.__class__.__name__,
                    content=coerce_dict_to_str(call.rets.return_values),
                    metadata_=self.metadata,
                    ts=call.perf.end_time,
                    call_idx=idx,
                )
            elif call.rets is not None:
                raise ValueError(f"Cannot extract event from call returns: {call.rets}")
        if record.main_error is not None and record.main_error != "None":
            yield MessageInfo(
                source=MessageSource.SYSTEM,
                label="Error",
                content=coerce_dict_to_str(record.main_error),
                metadata_=self.metadata,
                ts=record.perf.end_time,
                content_type=MessageContentType.JSON,
                call_idx=None,
            )


def jsonify_tool(tool: BaseTool, **opts) -> JSON:
    opts = opts or {}
    opts["redact_keys"] = True

    def _jsonify(x):
        return jsonify(x, **opts)

    j = {}

    if isinstance(tool, Tool) and (self := getattr(tool.func, "__self__", None)):
        j["tool_impl"] = {
            "method": tool.func.__name__,
            "class": fq_class_name(self.__class__),
            "instance": _jsonify(self),
        }

    j["tool"] = _jsonify(tool)
    return j


def fq_class_name(cls: Type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def coerce_dict_to_str(d: Union[dict, str], truncate_single_key: bool = True) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        if len(d) == 0:
            return ""
        if len(d) == 1 and truncate_single_key:
            return list(d.values())[0]
        return json.dumps(d)
    raise ValueError(f"Cannot coerce dict to str: {d}")
