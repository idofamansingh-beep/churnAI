import streamlit as st


st.set_page_config(page_title="ChurnAI", page_icon="🤖", layout="wide")

st.title("🤖 ChurnAI")
st.write("Run the maintained dashboard with:")
st.code(r".\venv\Scripts\streamlit.exe run .\app\dashboard\app.py", language="powershell")
st.markdown("[Open dashboard source](app/dashboard/app.py)")
