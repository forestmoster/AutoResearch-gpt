import openai
import streamlit as st
from streamlit_chat import message
import os


def num_text(text: str)-> int:
    num=len(text)
    return num



openai_api_key = os.getenv('OPENAI_API_KEY')
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if "回答内容"not in st.session_state:
    st.session_state["回答内容"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if '回答次数' not in st.session_state:
    st.session_state['回答次数'] = 1




styl = """
<style>
    .stTextInput {
        position: fixed;
        bottom: 2rem;
        background-color: white;
        right:700  
        left:500;
        border-radius: 36px; 
        z-index:4;
    }
    .stButton{
        position: fixed;
        bottom: 2rem;
        left:500; 
        right:500;
        z-index:4;
    }

    @media screen and (max-width: 1000px) {
        .stTextInput {
            left:2%; 
            width: 100%;
            bottom: 2.1rem;  
            z-index:2; 
        }                        
        .stButton {            
            left:2%;  
            width: 100%;       
            bottom:0rem;
            z-index:3; 
        }          
    } 

</style>

"""

st.markdown(styl, unsafe_allow_html=True)



st.title("💬 chartgpt")
st.caption('此为chatgpt3.5原始版本')
# openai.api_key = st.secrets.openai_api_key
if "messages_game" not in st.session_state:
    st.session_state["messages_game"] = [{"role": "assistant", "content":"现在问我一个问题"}]
if "回答内容_game" not in st.session_state:
    st.session_state["回答内容_game"] = [{"role": "system", "content": "尽量详细的回答用户的问题"},
                                         {"role": "assistant","content": "现在问我一个问题"}]


with st.form("my_form", clear_on_submit=True):
    st.header("🎈欢迎🎉🎉🎉用户🎉🎉🎉使用🎈")
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="写出你的问题",
        label_visibility="collapsed",max_chars=500
    )
    b.form_submit_button("Send", use_container_width=True)

i=0
for msg in st.session_state["messages_game"]:
    i=i+1
    message(message=msg["content"], is_user=msg["role"] == "user", key=f"message{i}")


if user_input :
    openai.api_key = openai_api_key
    st.session_state["messages_game"].append({"role": "user", "content": user_input})
    st.session_state["回答内容_game"].append({"role": "user", "content": user_input})
    message(user_input, is_user=True)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages= st.session_state["回答内容_game"],
                                            temperature=0.5,)
    msg = response.choices[0].message
    st.session_state["messages_game"].append(msg)
    st.session_state["回答内容_game"].append(msg)
    st.session_state['回答次数']=st.session_state['回答次数']+1
    message(msg.content)

    conversation_string = ""
    short_state_num=len(st.session_state["回答内容_game"])
    start_round = int(short_state_num*3/10)
    end_round = int(short_state_num*7/10)

    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_game"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num=num_text(conversation_string)
    if conversation_string_num >5200 or st.session_state['回答次数'] > 15:
        del st.session_state["回答内容_game"][start_round : end_round]
        st.session_state['回答次数'] = 1












