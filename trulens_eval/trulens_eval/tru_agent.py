"""
# Basic input output instrumentation and monitoring.
"""
import json
import logging
from typing import Optional, Dict, Union, Iterable, Type

from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.schema import AgentFinish, AgentAction
from langchain.tools import BaseTool, Tool

from trulens_eval.instruments import InstrumentBuilder
from trulens_eval.schema import MessageInfo, MessageSource, Record, MessageContentType
from trulens_eval.sessions import SessionableApp, start_session, MessagesExtractor
from trulens_eval.util import jsonify, JSON

logger = logging.getLogger(__name__)


class TruAgent(SessionableApp):
    app: AgentExecutor

    def __init__(self, app: AgentExecutor, instrument_builder: Optional[InstrumentBuilder] = None, **kwargs):
        instrument_builder = instrument_builder or InstrumentBuilder()
        instrument_builder.with_obj_methods(app.agent, "plan", "aplan")

        for tool in app.tools:
            instrument_builder.with_obj_methods(tool, "_run", "_arun")

        super().__init__(app, instrument_builder, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_with_session(*args, **kwargs)[0]

    def call_with_session(self, inputs: dict, metadata: Optional[Dict[str, str]] = None, **kwargs):
        with start_session(self, metadata) as session_runner:
            out, rec = session_runner.call_with_record(inputs=inputs, metadata=metadata, **kwargs)
            return out, rec, session_runner.session


class OpenAiFunctionsMessagesExtractor(MessagesExtractor):
    metadata: Dict[str, str]
    tools: Dict[str, dict]

    def __init__(self, agent: AgentExecutor, metadata: Optional[Dict[str, str]] = None, **kwargs):
        metadata = metadata or {}
        metadata["agent_type"] = AgentType.OPENAI_FUNCTIONS
        tools = {tool.name: jsonify_tool(tool) for tool in agent.tools}
        super().__init__(metadata=metadata, tools=tools, **kwargs)

    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[MessageInfo]:
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
                )
            if steps := call.args["intermediate_steps"]:
                yield MessageInfo(
                    source=MessageSource.SYSTEM,
                    label="Observation",
                    content=steps[-1][1],
                    metadata_=self.metadata,
                    ts=call.perf.start_time,
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
                )
            elif call.rets is not None:
                raise ValueError(f"Cannot extract event from call returns: {call.rets}")
        if error:
            yield MessageInfo(
                source=MessageSource.SYSTEM,
                label="Error",
                content=repr(error),
                metadata_=self.metadata,
                ts=record.perf.end_time,
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
