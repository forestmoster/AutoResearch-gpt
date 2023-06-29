import openai
import streamlit as st
from streamlit_chat import message
import os


def num_text(text: str)-> int:
    num=len(text)
    return num



openai_api_key = os.getenv('OPENAI_API_KEY')

if '回答次数_outline' not in st.session_state:
    st.session_state['回答次数_outline'] = 1




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



st.title("💬 烟台南山学院编写教学大纲")
st.caption('严格按照格式编写大纲！！！')
# openai.api_key = st.secrets.openai_api_key
if "messages_outline" not in st.session_state:
    st.session_state["messages_outline"] = [{"role": "assistant", "content":"现在开始协助老师编写教学大纲"}]
if "回答内容_outline" not in st.session_state:
    st.session_state["回答内容_outline"] = [{"role": "system", "content": '''现在开始协助老师，按照科目编写教学大纲，每一个大纲的字数在1000字左右,大纲模板如下'
"《xxx》课程教学大纲
课程名称：    	       课程代码：
课程类型: 专业必修课程
学　　分：4	总学时：64	理论学时：64 实验（上机）学时：	
先修课程：     适用专业：环境设计
一、课程性质、目的和任务 
1、课程性质：《xxx》课程是环境设计专业必修课程。
2、课程目的和任务：
二、教学基本要求
1、 知识、能力、素质的基本要求；
2、 教学模式基本要求：　
3、教学要求：
1）通过讲授，使学生初步了解手工印染的发展概况及艺术特征。
2）通过讲授和实际操作，使学生初步掌握手工印染的制作过程和表现方法。
3）掌握一定的构思、创作方法。
三、教学内容及要求
第1章　
了解、理解：
掌握：
第2章　
了解、理解：
掌握：
第3章　
了解、理解：
掌握：
第4章　
了解、理解：
掌握：
第5章　
了解、理解：
掌握：
四、实验（上机）内容
无
五、学时分配 
序号	课程内容	教 学 时 数
		讲 授	习题课	实  验	小  计
1	第1章　染织技术的发展历程	10			10
2	第2章　国内外染织艺术	8			8
3	第3章　染织图案风格	6			6
4	第4章　染织图案设计	16			16
5	第5章　染织技法及工艺	24			24
合  计	64			64
六、考核办法
1.考核方式：考查
2.成绩评定：本课程按照百分制模式进行考核，课程综合成绩总计满分为100分，其中平时考核成绩占30%、实践或技能综合考核成绩占70%。
（1）平时成绩主要考核学生考勤、平时作业、随堂提问等情况，占总成绩的30%；
（2）综合成绩根据学生课程作业质量进行考核评定，占总成绩的70%。
七、推荐教材和教学参考书  
教  材：
参考书：
教材和参考书要求近5年
八、说明     
制订：环境设计教研室
审定：刘霞
批准：赵君超
'''},
                                         {"role": "assistant",
                                          "content": "现在开始协助老师编写教学大纲"}]


with st.form("my_form", clear_on_submit=True):
    st.header("🎈开始🎉🎉🎉大纲🎉🎉🎉编写🎈")
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="写出你的要求",
        label_visibility="collapsed",max_chars=500
    )
    b.form_submit_button("Send", use_container_width=True)

i=0
for msg in st.session_state["messages_outline"]:
    i=i+1
    message(message=msg["content"], is_user=msg["role"] == "user", key=f"message{i}")


if user_input :
    openai.api_key = openai_api_key
    st.session_state["messages_outline"].append({"role": "user", "content": user_input})
    st.session_state["回答内容_outline"].append({"role": "user", "content": user_input})
    message(user_input, is_user=True)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages= st.session_state["回答内容_outline"],
                                            temperature=0.5,)
    msg = response.choices[0].message
    st.session_state["messages_outline"].append(msg)
    st.session_state["回答内容_outline"].append(msg)
    st.session_state['回答次数_outline']=st.session_state['回答次数_outline']+1
    message(msg.content)

    conversation_string = ""
    short_state_num=len(st.session_state["回答内容_outline"])
    start_round = int(short_state_num*3/10)
    end_round = int(short_state_num*7/10)

    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_outline"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num=num_text(conversation_string)
    if conversation_string_num >3200 or st.session_state['回答次数_outline'] > 15:
        del st.session_state["回答内容_outline"][start_round : end_round]
        st.session_state['回答次数_outline'] = 1