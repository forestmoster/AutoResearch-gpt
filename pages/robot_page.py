import openai
import streamlit as st
from streamlit_chat import message
import os
import ask_page
GPT_MODEL = "gpt-3.5-turbo"
st.title("💬 烟台南山学院 GPT")
# openai.api_key = st.secrets.openai_api_key
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好同学有什么可以帮助你的"}]
with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="What would you like to say?",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

for msg in st.session_state.messages:
    message(msg["content"], is_user=msg["role"] == "user",avatar_style="big-ears-neutral")

if user_input:
    openai.api_key = openai_api_key=os.getenv('OPENAI_API_KEY')
    st.session_state.messages.append({"role": "user", "content": user_input})
    message(user_input, is_user=True)
    response = ask_page.ask(query=st.session_state.messages, model=GPT_MODEL, token_budget=2000 - 500)
    st.session_state.messages.append(msg)
    message(msg.content)