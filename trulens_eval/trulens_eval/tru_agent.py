"""
# Basic input output instrumentation and monitoring.
"""
import json
import logging
from typing import Optional, Dict, Union, Iterable

from langchain.agents import AgentExecutor
from langchain.agents import AgentType
from langchain.schema import AgentFinish, AgentAction

from trulens_eval.schema import MessageInfo, MessageSource, Record
from trulens_eval.sessions import SessionableApp, start_session, MessagesExtractor
from trulens_eval.util import jsonify

logger = logging.getLogger(__name__)


class TruAgent(SessionableApp):
    app: AgentExecutor

    def __init__(self, app: AgentExecutor, **kwargs):
        instrument_conf = {
            "modules": {app.agent.__module__},
            "classes": {app.agent.__class__},
            "methods": {
                m: lambda o: isinstance(o, app.agent.__class__)
                for m in ["plan", "aplan"]
            },
        }
        super().__init__(app, instrument_conf, **kwargs)

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
        tools = {tool.name: jsonify(tool) for tool in agent.tools}
        super().__init__(metadata=metadata, tools=tools, **kwargs)

    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[MessageInfo]:
        for idx, call in enumerate(record.calls):
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
                    content=coerce_dict_to_str(call.rets.tool_input),
                    metadata_={
                        **self.metadata,
                        "tool": self.tools.get(call.rets.tool)
                    },
                    ts=call.perf.end_time,
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


def coerce_dict_to_str(d: Union[dict, str]) -> str:
    if isinstance(d, str):
        return d
    if isinstance(d, dict):
        if len(d) == 0:
            return ""
        if len(d) == 1:
            return list(d.values())[0]
        return json.dumps(d)
    raise ValueError(f"Cannot coerce dict to str: {d}")
