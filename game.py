import openai
import streamlit as st
from streamlit_chat import message
import os


openai_api_key = os.getenv('OPENAI_API_KEY')



st.title("💬 烟台南山学院冒险游戏")
st.caption('是2023年9月的刚进学校的新生，无意间发现了烟台南山学院关于外星人的秘密，让我们开始探索吧！！！')
# openai.api_key = st.secrets.openai_api_key
if "messages" not in st.session_state:
    st.session_state["messages_game"] = [{"role": "assistant", "content":"现在选择一个专业，让我们开始冒险吧！！"}]
if "回答内容"not in st.session_state:
    st.session_state["回答内容_game"] = [{"role": "assistant", "content": "现在让我们我玩一个游戏,主人公是2023年9月的刚进学校的还没有选专业的新生，无意中发现了隐藏在学校后面的有关外星人的惊人秘密。"
                                                                     "现在开始选择专业并且进行冒险"
                                                                     "你提供两个选项，我来选，然后进行回答。"
                                                                     "故事背景是烟台南山学院，烟台南山学院有三个校区，东海校区（东海校区靠海，面积比较大，科技与数据学院、智能科学与工程学院、"
                                                                     "材料科学与工程学院、纺织与服装学院、化学工程与技术学院、航空科学与工程学院、健康学院、经济与管理学院，马克思主义学院。），"
                                                                     "南山校区（在音乐校区山脚下，国学与外语学院、艺术与设计学院），"
                                                                     "音乐校区（在山上，靠近南山旅游景区，音乐与舞蹈学院)。"}]

with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="做出你的选择",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

if st.button('重新开始一个冒险'):
    del st.session_state["回答内容_game"]
    del st.session_state["messages_game"]
    st.session_state["messages_game"] = [{"role": "assistant", "content":"现在选择一个专业，让我们开始冒险吧！！！"}]
    st.session_state["回答内容_game"] = [{"role": "assistant", "content": "现在让我们我玩一个游戏,主人公是2023年9月的刚进学校的还没有选专业的新生，无意中发现了隐藏在学校后面的有关外星人的惊人秘密。"
                                                                     "现在开始选择专业并且进行冒险"
                                                                     "你提供两个选项，我来选，然后进行回答。"
                                                                     "故事背景是烟台南山学院，烟台南山学院有三个校区，东海校区（东海校区靠海，面积比较大，科技与数据学院、智能科学与工程学院、"
                                                                     "材料科学与工程学院、纺织与服装学院、化学工程与技术学院、航空科学与工程学院、健康学院、经济与管理学院，马克思主义学院。），"
                                                                     "南山校区（在音乐校区山脚下，国学与外语学院、艺术与设计学院），"
                                                                     "音乐校区（在山上，靠近南山旅游景区，音乐与舞蹈学院)。"}]
    # 清空文本输入框的内容
    user_input = ""


i=0
for msg in st.session_state.messages_game:
    i=i+1
    message(message=msg["content"], is_user=msg["role"] == "user", key=f"message{i}")

if user_input and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

if user_input and openai_api_key:
    openai.api_key = openai_api_key
    st.session_state.messages_game.append({"role": "user", "content": user_input})

    message(user_input, is_user=True)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages= st.session_state["回答内容_game"])
    msg = response.choices[0].message
    st.session_state.messages_game.append(msg)
    st.session_state["回答内容_game"].append(msg)
    message(msg.content)
