import openai
import streamlit as st
from streamlit_chat import message
import os
from ask_page import  query_message

openai_api_key = os.getenv('OPENAI_API_KEY')

st.title("💬 烟台南山学院ai助手")
st.caption('你可以查询有关南山学院的问题，也可以帮助我完善数据库，上传:blue[docx文件]')
# openai.api_key = st.secrets.openai_api_key
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if "回答内容"not in st.session_state:
    st.session_state["回答内容"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if '回答次数' not in st.session_state:
    st.session_state['回答次数'] = 1

with st.button('重新开始一个回答'):
    del st.session_state['回答内容']
    del st.session_state['messages']
    st.session_state["回答内容"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]


with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="在这打字，回答问题",
        label_visibility="collapsed",max_chars=500
    )
    b.form_submit_button("Send", use_container_width=True)


st.session_state
i=0
for msg in st.session_state.messages:
    i=i+1
    message(message=msg["content"], is_user=msg["role"] == "user", key=f"message{i}")



query_message=query_message(query=user_input,token_budget=2000 - 500,)


if user_input :
    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state['回答内容'].append({"role": "user", "content": query_message})
    message(user_input, is_user=True)

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages= st.session_state['回答内容'])
    # response = ask_page.ask_robot(query=query_response, model="gpt-3.5-turbo", token_budget=2000 - 500)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.session_state['回答内容'].append(msg)
    st.session_state['回答次数'] = st.session_state['回答次数'] + 1
    # 修改 st.session_state['回答内容'] 中的最后一条消息的内容
    st.session_state['回答内容'][-2]["content"] = user_input
    message(msg.content)
st.session_state
