import json
import logging
import warnings
from datetime import datetime
from typing import List, Tuple, Sequence, Optional, Iterable, Union

import numpy as np
import pandas as pd
from pydantic import Field
from sqlalchemy import Engine, create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from trulens_eval import schema
from trulens_eval.database import orm
from trulens_eval.database.migrations import upgrade_db
from trulens_eval.database.utils import for_all_methods, run_before, is_legacy_sqlite, is_memory_sqlite, \
    check_db_revision, migrate_legacy_sqlite
from trulens_eval.db import DB
from trulens_eval.db_migration import MIGRATION_UNKNOWN_STR
from trulens_eval.schema import RecordID, FeedbackResultID, FeedbackDefinitionID, FeedbackResultStatus
from trulens_eval.util import JSON

logger = logging.getLogger(__name__)


@for_all_methods(
    run_before(lambda self, *args, **kwargs: check_db_revision(self.engine)),
    _except=["migrate_database", "reload_engine"]
)
class SqlAlchemyDB(DB):
    engine_params: dict = Field(default_factory=dict)
    session_params: dict = Field(default_factory=dict)
    engine: Engine = None
    Session: sessionmaker = None

    class Config:
        arbitrary_types_allowed: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reload_engine()
        if is_memory_sqlite(self.engine):
            warnings.warn(UserWarning(
                "SQLite in-memory may not be threadsafe. "
                "See https://www.sqlite.org/threadsafe.html"
            ))

    def reload_engine(self):
        self.engine = create_engine(**self.engine_params)
        self.Session = sessionmaker(self.engine, **self.session_params)

    def insert_session(self, session: schema.Session):
        with self.Session.begin() as __session__:
            __session__.add(orm.Session.parse(session))

    def update_session(self, session_id: schema.SessionID, **kwargs):
        with self.Session.begin() as __session__:
            _session = __session__.query(orm.Session).filter_by(session_id=session_id)
            for key, val in kwargs.items():
                setattr(_session, key, val)

    def get_sessions(self) -> List[str]:
        with self.Session.begin() as __session__:
            stmt = select(orm.Session.session_id)
            return [row[0] for row in __session__.execute(stmt)]

    def insert_messages(self, messages: Iterable[schema.Message]):
        with self.Session.begin() as session:
            session.add_all((orm.Message.parse(e) for e in messages))

    def get_messages(self, session_id: schema.SessionID) -> pd.DataFrame:
        with self.Session.begin() as __session__:
            _session = __session__.query(orm.Session).filter_by(session_id=session_id).one()
            cols = ["ts", "source", "label", "content", "metadata_"]
            df = pd.DataFrame(
                data=([getattr(e, c) for c in cols] for e in _session.messages),
                columns=cols,
            )
            df["ts"] = df["ts"].apply(datetime.fromtimestamp)
            df["metadata_"] = df["metadata_"].apply(json.loads)
            return df.sort_values("ts")

    @classmethod
    def from_db_url(cls, url: str) -> "SqlAlchemyDB":
        return cls(engine_params={"url": url})

    def migrate_database(self):
        """Migrate database schema to the latest revision"""
        if is_legacy_sqlite(self.engine):
            migrate_legacy_sqlite(self.engine)
        else:
            upgrade_db(self.engine, revision="head")
        self.reload_engine()  # let sqlalchemy recognize the migrated schema

    def reset_database(self):
        raise NotImplementedError(
            f"Resetting the database is not implemented for `{self.__class__}`. "
            "Please perform this operation by connecting to the database directly"
        )

    def insert_record(self, record: schema.Record) -> schema.RecordID:
        with self.Session.begin() as session:
            _record = orm.Record.parse(record)
            session.add(_record)
            return _record.record_id

    def get_app(self, app_id: str) -> Optional[JSON]:
        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition).filter_by(app_id=app_id).first():
                return json.loads(_app.app_json)

    def insert_app(self, app: schema.AppDefinition) -> schema.AppID:
        with self.Session.begin() as session:
            if _app := session.query(orm.AppDefinition).filter_by(app_id=app.app_id).first():
                _app.app_json = app.json()
            else:
                _app = orm.AppDefinition.parse(app)
                session.add(_app)
            return _app.app_id

    def insert_feedback_definition(self, feedback_definition: schema.FeedbackDefinition) -> schema.FeedbackDefinitionID:
        with self.Session.begin() as session:
            if _fb_def := session.query(orm.FeedbackDefinition) \
                    .filter_by(feedback_definition_id=feedback_definition.feedback_definition_id) \
                    .first():
                _fb_def.app_json = feedback_definition.json()
            else:
                _fb_def = orm.FeedbackDefinition.parse(feedback_definition)
                session.add(_fb_def)
            return _fb_def.feedback_definition_id

    def get_feedback_defs(self, feedback_definition_id: Optional[str] = None) -> pd.DataFrame:
        with self.Session.begin() as session:
            q = select(orm.FeedbackDefinition)
            if feedback_definition_id:
                q = q.filter_by(feedback_definition_id=feedback_definition_id)
            fb_defs = (row[0] for row in session.execute(q))
            return pd.DataFrame(
                data=((fb.feedback_definition_id, json.loads(fb.feedback_json)) for fb in fb_defs),
                columns=["feedback_definition_id", "feedback_json"],
            )

    def insert_feedback(self, feedback_result: schema.FeedbackResult) -> schema.FeedbackResultID:
        with self.Session.begin() as session:
            _feedback_result = orm.FeedbackResult.parse(feedback_result)
            session.add(_feedback_result)
            return _feedback_result.feedback_result_id

    def get_feedback(
        self,
        record_id: Optional[RecordID] = None,
        feedback_result_id: Optional[FeedbackResultID] = None,
        feedback_definition_id: Optional[FeedbackDefinitionID] = None,
        status: Optional[Union[FeedbackResultStatus, Sequence[FeedbackResultStatus]]] = None,
        last_ts_before: Optional[datetime] = None
    ) -> pd.DataFrame:
        with self.Session.begin() as session:
            q = select(orm.FeedbackResult)
            if record_id:
                q = q.filter_by(record_id=record_id)
            if feedback_result_id:
                q = q.filter_by(feedback_result_id=feedback_result_id)
            if feedback_definition_id:
                q = q.filter_by(feedback_definition_id=feedback_definition_id)
            if status:
                if isinstance(status, FeedbackResultStatus):
                    status = [status.value]
                q = q.filter(orm.FeedbackResult.status.in_([s.value for s in status]))
            if last_ts_before:
                q = q.filter(orm.FeedbackResult.last_ts < last_ts_before.timestamp())
            results = (row[0] for row in session.execute(q))
            return _extract_feedback_results(results)

    def get_records_and_feedback(self, app_ids: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Sequence[str]]:
        with self.Session.begin() as session:
            stmt = select(orm.AppDefinition)
            if app_ids:
                stmt = stmt.where(orm.AppDefinition.app_id.in_(app_ids))
            apps = (row[0] for row in session.execute(stmt))
            return AppsExtractor().get_df_and_cols(apps)


def _extract_feedback_results(results: Iterable[orm.FeedbackResult]) -> pd.DataFrame:
    def _extract(_result: orm.FeedbackResult):
        app_json = json.loads(_result.record.app.app_json)
        _type = schema.AppDefinition(**app_json).root_class
        return (
            _result.record_id, _result.feedback_result_id, _result.feedback_definition_id,
            _result.last_ts, FeedbackResultStatus(_result.status), _result.error, _result.name,
            _result.result, _result.cost_json, json.loads(_result.record.perf_json),
            json.loads(_result.calls_json)["calls"], json.loads(_result.feedback_definition.feedback_json),
            json.loads(_result.record.record_json), app_json, _type,
        )

    df = pd.DataFrame(
        data=(_extract(r) for r in results),
        columns=[
            'record_id', 'feedback_result_id', 'feedback_definition_id',
            'last_ts', 'status', 'error', 'fname',
            'result', 'cost_json', 'perf_json',
            'calls_json', 'feedback_json',
            'record_json', 'app_json', "type",
        ],
    )
    df["latency"] = _extract_latency(df["perf_json"])
    df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
    return df


def _extract_latency(perf_json: pd.Series) -> pd.Series:
    perf = perf_json.apply(
        lambda p: schema.Perf.parse_raw(p) if isinstance(p, str) else schema.Perf(**p)
        if p != MIGRATION_UNKNOWN_STR else MIGRATION_UNKNOWN_STR
    )
    return perf.apply(
        lambda p: p.latency.seconds
        if p != MIGRATION_UNKNOWN_STR else MIGRATION_UNKNOWN_STR
    )


def _extract_tokens_and_cost(cost_json: pd.Series) -> pd.DataFrame:
    def _extract(_cost_json: Union[str, dict]) -> Tuple[int, float]:
        if isinstance(_cost_json, str):
            _cost_json = json.loads(_cost_json)
        cost = schema.Cost(**_cost_json)
        return cost.n_tokens, cost.cost
    return pd.DataFrame(
        data=(_extract(c) for c in cost_json),
        columns=["total_tokens", "total_cost"],
    )


class AppsExtractor:
    app_cols = ["app_id", "app_json", "type"]
    rec_cols = ["record_id", "input", "output", "tags", "record_json", "cost_json", "perf_json", "ts"]
    extra_cols = ["latency", "total_tokens", "total_cost"]
    all_cols = app_cols + rec_cols + extra_cols

    def __init__(self):
        self.feedback_columns = set()

    def get_df_and_cols(self, apps: Iterable[orm.AppDefinition]) -> Tuple[pd.DataFrame, Sequence[str]]:
        df = pd.concat(self.extract_apps(apps))
        df["latency"] = _extract_latency(df["perf_json"])
        df.reset_index(drop=True, inplace=True)  # prevent index mismatch on the horizontal concat that follows
        df = pd.concat([df, _extract_tokens_and_cost(df["cost_json"])], axis=1)
        return df, list(self.feedback_columns)

    def extract_apps(self, apps: Iterable[orm.AppDefinition]) -> Iterable[pd.DataFrame]:
        yield pd.DataFrame([], columns=self.app_cols + self.rec_cols)  # prevent empty iterator
        for _app in apps:
            if _recs := _app.records:
                df = pd.concat(self.extract_records(_recs))

                for col in self.app_cols:
                    if col == "type":
                        df[col] = str(schema.AppDefinition.parse_raw(_app.app_json).root_class)
                    else:
                        df[col] = getattr(_app, col)

                yield df

    def extract_records(self, records: Iterable[orm.Record]) -> Iterable[pd.DataFrame]:
        for _rec in records:
            df = pd.DataFrame(self.extract_results(_rec.feedback_results), columns=["key", "value"]) \
                .pivot_table(columns="key", values="value", aggfunc=self.agg_result_or_calls) \
                .reset_index(drop=True).rename_axis("", axis=1)

            for col in self.rec_cols:
                df[col] = datetime.fromtimestamp(_rec.ts).isoformat() if col == "ts" else getattr(_rec, col)

            yield df

    def extract_results(self, results: Iterable[orm.FeedbackResult]) -> Iterable[Tuple[str, Union[float, dict]]]:
        for _res in results:
            self.feedback_columns.add(_res.name)
            yield _res.name, _res.result
            yield f"{_res.name}_calls", json.loads(_res.calls_json)["calls"]

    @classmethod
    def agg_result_or_calls(cls, *args):
        if not args:
            return None
        elif isinstance(args[0], dict):
            return args
        else:
            return np.mean(args)