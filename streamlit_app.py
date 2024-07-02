import streamlit as st
from streamlit_gsheets import GSheetsConnection

conn = st.connection("gsheets", type=GSheetsConnection)

df = conn.read(
    worksheet="tracker",
    ttl=60,
    )

st.dataframe(df)