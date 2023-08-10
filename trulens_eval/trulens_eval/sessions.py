import logging
from datetime import datetime
from typing import Iterable, Dict

import pandas as pd

from trulens_eval.app import App
from trulens_eval.messages import MessagesExtractor
from trulens_eval.schema import Session, Record, Message

logger = logging.getLogger(__name__)


class SessionManager:

    def __init__(self, app: App, messages_extractor: MessagesExtractor):
        self.app = app
        self.messages_extractor = messages_extractor
        self.session_ids = []

    def start_session(self, metadata: Dict[str, str] = None) -> "SessionRunner":
        session = Session(app_id=self.app.app_id, metadata_=metadata or {})
        self.session_ids.append(session.session_id)
        self.app.db.insert_session(session)  # TODO: make async with TP
        return SessionRunner(self, session)

    def end_session(self, session: Session):
        self.app.db.update_session(session.session_id, end_ts=datetime.now().timestamp())  # TODO: make async with TP

    def get_messages(self) -> pd.DataFrame:
        return self.app.db.get_messages(*self.session_ids)


class SessionRunner:

    def __init__(self, session_manager: SessionManager, session: Session):
        self.session_manager = session_manager
        self.session = session

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session_manager.end_session(self.session)

    def __call__(self, *args, **kwargs):
        return self.call_with_record(*args, **kwargs)[0]

    def get_messages(self) -> pd.DataFrame:
        return self.session_manager.app.db.get_messages(self.session.session_id)

    def call_with_record(self, *args, **kwargs):
        out, rec = self.session_manager.app.call_with_record(*args, **kwargs)
        self._handle_record(rec)  # TODO: make async with TP
        return out, rec

    def _handle_record(self, record: Record):
        self.session_manager.app.db.insert_messages(self._extract_messages(record))

    def _extract_messages(self, record: Record) -> Iterable[Message]:
        for msg_info in self.session_manager.messages_extractor(record):
            yield Message(
                record_id=record.record_id,
                session_id=self.session.session_id,
                **msg_info.dict()
            )
