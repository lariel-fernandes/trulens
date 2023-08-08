"""
# Basic input output instrumentation and monitoring.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Sequence, Iterable, Optional, Any, Union

from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import AgentAction
from langchain.schema import AgentFinish
from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Cost, Record, EventInfo, Event
from trulens_eval.schema import RecordAppCall
from trulens_eval.util import Class, SerialModel
from trulens_eval.util import FunctionOrMethod

logger = logging.getLogger(__name__)


class EventsExtractor(SerialModel, ABC):

    @abstractmethod
    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[EventInfo]:
        raise NotImplementedError()


class OpenAiFunctionsEventExtractor(EventsExtractor):

    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[EventInfo]:
        for idx, call in enumerate(record.calls):
            if idx == 0:
                yield EventInfo(
                    category="Input",
                    content=coerce_dict_to_str(call.args["kwargs"]),
                )
            if steps := call.args["intermediate_steps"]:
                yield EventInfo(
                    category="Observation",
                    content=steps[-1][1],
                )
            if isinstance(call.rets, AgentAction):
                yield EventInfo(
                    category=f"Action[{call.rets.tool}]",
                    content=coerce_dict_to_str(call.rets.tool_input),
                )
            elif isinstance(call.rets, AgentFinish):
                yield EventInfo(
                    category=call.rets.__class__.__name__,
                    content=coerce_dict_to_str(call.rets.return_values),
                )
            else:
                raise ValueError(f"Cannot extract event from call returns: {call.rets}")


class AppWithEvents(App):

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(AppWithEvents.call_with_record),
        const=True
    )

    events_extractor: EventsExtractor

    def __init__(self, app: Any, instrument_conf: dict, **kwargs):
        super().update_forward_refs()
        kwargs['app'] = app
        kwargs['root_class'] = Class.of_object(app)
        kwargs['instrument'] = Instrument(root_methods={self.call_with_record}, **instrument_conf)
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.call_with_record(*args, **kwargs)[0]

    def call_with_record(self, inputs: dict, **kwargs):
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

        return ret, ret_record

    def _extract_events(self, record: Record, error: Optional[Exception] = None) -> Iterable[Event]:
        return (
            Event(record_id=record.record_id, idx=idx + 1, **event_info.dict())
            for idx, event_info in enumerate(self.events_extractor(record, error))
        )

    def _handle_record(self, record: Record):
        super()._handle_record(record)
        events = self._extract_events(record)
        self._handle_events(events)

    def _handle_error(self, record: Record, error: Exception):
        super()._handle_error(record, error)
        events = self._extract_events(record, error)
        self._handle_events(events)

    def _handle_events(self, events: Iterable[Event]):
        self.db.insert_events(events)  # noqa


class TruAgent(AppWithEvents):
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
