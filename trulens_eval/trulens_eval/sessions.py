import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Optional, Any, Sequence, Iterable, Dict

import pandas as pd
from pydantic import Field

from trulens_eval.app import App
from trulens_eval.instruments import Instrument
from trulens_eval.provider_apis import Endpoint
from trulens_eval.schema import Session, RecordAppCall, Cost, Record, Message, MessageInfo, SessionID
from trulens_eval.util import FunctionOrMethod, Class, SerialModel

logger = logging.getLogger(__name__)


class MessagesExtractor(SerialModel, ABC):

    @abstractmethod
    def __call__(self, record: Record, error: Optional[Exception]) -> Iterable[MessageInfo]:
        raise NotImplementedError()


class SessionableApp(App):

    root_callable: ClassVar[FunctionOrMethod] = Field(
        default_factory=lambda: FunctionOrMethod.of_callable(SessionableApp.call_with_record),
        const=True
    )

    messages_extractor: Optional[MessagesExtractor] = None

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

    def __call__(self, *args, **kwargs):
        return self.call_with_record(*args, **kwargs)[0]

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

        if self.messages_extractor is not None:
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


class SessionRunner:

    def __init__(self, session: Session, app: SessionableApp):
        self.session = session
        self.app = app

    @property
    def session_id(self) -> SessionID:
        return self.session.session_id

    def get_messages(self) -> pd.DataFrame:
        return self.app.db.get_messages(self.session_id)

    def __call__(self, *args, **kwargs):
        return self.call_with_record(*args, **kwargs)[0]

    def call_with_record(self, inputs: dict, **kwargs):
        return self.app.call_with_record(session=self.session, inputs=inputs, **kwargs)

    def __enter__(self):
        # TP().runlater(self.db.insert_session, session=self.session)
        self.app.db.insert_session(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TP().runlater(self.db.update_session, session_id=self.session.session_id, end_ts=datetime.now().timestamp())
        self.app.db.update_session(self.session.session_id, end_ts=datetime.now().timestamp())


def start_session(app: SessionableApp, metadata: Dict[str, str] = None) -> SessionRunner:
    """
    Usage:
        app = SessionableApp(
            chain_or_agent,  # implements: __call__(self, inputs: dict, **kwargs)
            app_id="...",
            messages_extractor=<some_impl_of_MessagesExtractor>
        )

        with start_session(app) as session_runner:
            for row in dataset:
                session_runner(inputs={...})  # each call produces messages in the same session

        df_messages = session_runner.get_messages()
    """
    return SessionRunner(
        session=Session(
            app_id=app.app_id,
            metadata_=metadata or {},
            start_ts=datetime.now(),
        ),
        app=app,
    )
