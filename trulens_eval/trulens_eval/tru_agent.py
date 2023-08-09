"""
# Basic input output instrumentation and monitoring.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Sequence, Iterable, Optional, Any, Union, Dict

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import AgentAction
from langchain.schema import AgentFinish
from pydantic import Field

from trulens_eval.app import App
from trulens_eval.db import DB
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost, Record, MessageInfo, Message, MessageSource, Session
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class, SerialModel
from trulens_eval.util import FunctionOrMethod

logger = logging.getLogger(__name__)


class MessagesExtractor(SerialModel, ABC):

    @abstractmethod
    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[MessageInfo]:
        raise NotImplementedError()


class OpenAiFunctionsMessagesExtractor(MessagesExtractor):
    metadata: Dict[str, str]

    def __init__(self, metadata: Optional[Dict[str, str]] = None):
        metadata = metadata or {}
        metadata["agent_type"] = "OPENAI_FUNCTIONS"
        super().__init__(metadata=metadata)

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
                    metadata_=self.metadata,
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


class SessionableApp(App):

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(SessionableApp.call_with_record),
        const=True
    )

    messages_extractor: MessagesExtractor

    def __init__(self, app: Any, instrument_conf: Optional[dict] = None, **kwargs):
        super().update_forward_refs()
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        instrument_conf = instrument_conf or {
            "modules": {app.__module__},
            "classes": {app.__class__},
            "methods": {"__call__": lambda o: isinstance(o, app.__class__)},
        }
        kwargs['instrument'] = Instrument(root_methods={self.call_with_record}, **instrument_conf)
        super().__init__(**kwargs)

    def call_with_record(self, session: Session, inputs: dict, **kwargs):
        # Wrapped calls will look this up by traversing the call stack
        record: Sequence[RecordAppCall] = []

        ret = None
        error = None

        cost: Cost = Cost()

        start_time = None

        try:
            start_time = datetime.now()
            ret, cost = Endpoint.track_all_costs_tally(
                lambda: self.app(inputs=inputs, **kwargs)
            )
            end_time = datetime.now()

        except BaseException as e:
            end_time = datetime.now()
            error = e
            logger.error(f"App raised an exception: {e}")

        assert len(record) > 0, "No information recorded in call."

        main_input = inputs
        if len(main_input) == 1:
            main_input = list(main_input.values())[0]

        ret_record_args = {"main_input": main_input}

        if ret is not None:
            ret_record_args['main_output'] = ret

        ret_record = self._post_record(
            ret_record_args, error, cost, start_time, end_time, record
        )

        # TP().runlater(
        #     self._post_messages, session=session, record=ret_record, error=error
        # )
        self._post_messages(session, ret_record, error)

        return ret, ret_record

    def _extract_messages(
        self, session: Session, record: Record, error: Optional[Exception] = None
    ) -> Iterable[Message]:
        for msg_info in self.messages_extractor(record, error):
            yield Message(
                record_id=record.record_id,
                session_id=session.session_id,
                **msg_info.dict()
            )

    def _post_messages(self, session: Session, record: Record, error: Optional[Exception] = None):
        self.db.insert_messages(self._extract_messages(session, record, error))  # noqa


class SessionController:

    def __init__(self, session: Session, db: DB):
        self.session = session
        self.db = db

    def __enter__(self):
        # TP().runlater(self.db.insert_session, session=self.session)
        self.db.insert_session(self.session)
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TP().runlater(self.db.update_session, session_id=self.session.session_id, end_ts=datetime.now().timestamp())
        self.db.update_session(self.session.session_id, end_ts=datetime.now().timestamp())


def start_session(app: SessionableApp, metadata: Dict[str, str] = None):
    return SessionController(
        session=Session(
            app_id=app.app_id,
            metadata_=metadata or {},
            start_ts=datetime.now(),
        ),
        db=app.db,
    )


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
        with start_session(self, metadata) as session:
            out, rec = self.call_with_record(session, inputs=inputs, metadata=metadata, **kwargs)
            return out, rec, session


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
