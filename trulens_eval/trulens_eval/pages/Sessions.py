from datetime import datetime

import streamlit as st
from ux.add_logo import add_logo

from trulens_eval import Tru
from trulens_eval.utils.text import UNICODE_GEAR

st.set_page_config(page_title="Sessions", layout="wide")

st.title("Sessions")

st.runtime.legacy_caching.clear_cache()

add_logo()

tru = Tru()
lms = tru.db

state = st.session_state


session_id = st.selectbox('Session ID', [""] + lms.get_sessions())

today = datetime.today().date()


def format_ts(_ts: datetime) -> str:
    year = f" {_ts.year}" if _ts.year != today.year else ""
    date = f"{_ts:%b} {_ts.day}{year}, " if _ts.date() != today else ""
    return f"{date}{_ts:%H}:{_ts:%M}:{_ts:%S}"


if session_id:
    df_msg = lms.get_messages(session_id)
    for _, row in df_msg.iterrows():
        with st.chat_message(row.source, avatar=UNICODE_GEAR if row.source == "system" else None):
            show_meta = st.button(f"**{row.label}** - *{format_ts(row.ts)}*")
            st.write(row.content)
            if show_meta:
                with st.expander("Details"):
                    st.write(row.metadata_)
