import json
from datetime import datetime

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
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
    st.subheader("Messages")
    df_msg = lms.get_messages(session_id)
    for idx, row in df_msg.iterrows():
        with st.chat_message(row.source, avatar=UNICODE_GEAR if row.source == "system" else None):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                show_meta = st.button(f"**{row.label}** - *{format_ts(row.ts)}*")
                st.write(json.loads(row.content) if row.content_type == "json" else row.content)
            if show_meta:
                with st.expander("Details"):
                    st.write(row.metadata_)
            with col2:
                if st.button("See record page ↗️", key=f"see-record-{idx}"):
                    st.session_state["selected_record_id"] = row.record_id
                    switch_page("Evaluations")
