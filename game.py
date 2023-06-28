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



st.title("💬 烟台南山学院ai文字游戏")
st.caption('你是2023年9月的刚进学校的新生，无意间发现了烟台南山学院关于宝藏的秘密，让我们开始探索吧！！！')
# openai.api_key = st.secrets.openai_api_key
if "messages_game" not in st.session_state:
    st.session_state["messages_game"] = [{"role": "assistant", "content":"现在选择一个校区和学院，让我们开始冒险吧！！！"}]
if "回答内容_game" not in st.session_state:
    st.session_state["回答内容_game"] = [{"role": "system", "content": "现在玩一个冒险的文字游戏,。每一次的语句都要提供1,2,3,4这种类型的选项，我来选，然后进行回答。每一次的语句要简短干练控制在6句以内\n\n"
                                                                       "故事简介：在烟台南山学院，主人公无意听闻了关于失落的校园宝藏的传说，随着冒险的深入，主人公一群人发现宝藏背后隐藏着与外星人相关的谜团。"
                                                                       "角色设定：我：我是2023年9月的刚进烟台南山学院的还没有选专业的新生，无意中发现了隐藏在学校后面的有关宝藏的惊人秘密。"                                                                                                                                                                                                
                                                                       "舍友：每个人都有各自的特长和技能，他们一起组成冒险队伍，共同寻找宝藏和解开外星人谜团。"
                                                                       "对手：神秘的学生会或其他学生，他们也对宝藏和外星人的谜团感兴趣，试图阻止主人公和朋友们的冒险。"
                                                                       "烟台南山学院管理严格，特别是对于学校卫生方面。有三个校区，东海校区（东海校区靠海，面积比较大，科技与数据学院、智能科学与工程学院、"
                                                                       "材料科学与工程学院、纺织与服装学院、化学工程与技术学院、航空科学与工程学院、健康学院、经济与管理学院，马克思主义学院。），"
                                                                       "南山校区（在音乐校区山脚下，国学与外语学院、艺术与设计学院），"
                                                                       "音乐校区（在山上，靠近南山旅游景区，音乐与舞蹈学院)。"},
                                         {"role": "assistant",
                                          "content": "现在选择你出生的校区和学院，然后开始冒险吧！！！"}]


with st.form("my_form", clear_on_submit=True):
    st.header("🎈欢迎🎉🎉🎉新同学🎉🎉🎉入学🎈")
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="做出你的选择",
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
                                            temperature=0.5,top_p=1.0,)
    msg = response.choices[0].message
    st.session_state["messages_game"].append(msg)
    st.session_state["回答内容_game"].append(msg)
    st.session_state['回答次数']=st.session_state['回答次数']+1
    message(msg.content)

    conversation_string = ""
    short_state_num=len(st.session_state["回答内容_game"])
    start_round = int(short_state_num*1/4)
    end_round = int(short_state_num*4/5)
    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_game"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num=num_text(conversation_string)
    st.write(conversation_string_num)


    if conversation_string_num >2300 or st.session_state['回答次数'] > 15:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=[{"role": "system",
                                                           "content": '请总结归纳以下的对话，将字数控制在200字以内'},
                                                          {"role": "user", "content": conversation_string}]
                                                )
        summary = response.choices[0].message["content"]
        st.session_state["回答内容_game"][start_round : end_round] = [
            {"role": "assistant", "content": summary}
        ]
        st.session_state['回答次数'] = 1
        st.write(st.session_state["回答内容_game"])










