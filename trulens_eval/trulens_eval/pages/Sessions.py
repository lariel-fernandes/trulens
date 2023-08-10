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


session_id = st.selectbox('Session ID', [""] + lms.get_session_ids())

today = datetime.today().date()


def format_ts(_ts: datetime) -> str:
    year = f" {_ts.year}" if _ts.year != today.year else ""
    date = f"{_ts:%b} {_ts.day}{year}, " if _ts.date() != today else ""
    return f"{date}{_ts:%H}:{_ts:%M}:{_ts:%S}"


if session_id:
    st.subheader("Messages")
    df_session = lms.get_messages(session_id)
    for rec_id, df_msg in df_session.groupby("record_id", sort=False):
        col1, col2 = st.columns(2, gap="large")
        with col2:
            if st.button(f"Record ID: {rec_id} ↗️", key=f"see-record-{rec_id}"):
                st.session_state["selected_record_id"] = rec_id
                switch_page("Evaluations")
        with col1:
            for _, row in df_msg.iterrows():
                with st.chat_message(row.source, avatar=UNICODE_GEAR if row.source == "system" else None):
                    show_meta = st.button(f"**{row.label}** - *{format_ts(row.msg_ts)}*")
                    st.write(json.loads(row.content) if row.content_type == "json" else row.content)
                    if show_meta:
                        with st.expander("Details"):
                            st.write(row.metadata_)
        st.markdown("---")
