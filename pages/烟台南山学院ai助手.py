from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser, BaseMultiActionAgent, initialize_agent, BaseSingleActionAgent,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain, PromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from langchain.tools import tool
import os
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.tools import DuckDuckGoSearchRun, Tool
import ask_page

styl = """
<style>

    .stButton{
        position: fixed;
        bottom: 1rem;
        left:500;
        right:500;
        z-index:999;
    }

    @media screen and (max-width: 1000px) {

        .stButton {
            left:2%;
            width: 100%;
            bottom:1.1rem;
            z-index:999;
        }
    }

</style>

"""

st.markdown(styl, unsafe_allow_html=True)

st.title("💬 烟台南山学院ai助手")
st.caption('你可以查询有关南山学院的问题，也可以帮助我完善数据库，上传:blue[文件]')

if "messages_web" not in st.session_state:
    st.session_state["messages_web"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if "回答内容_web" not in st.session_state:
    st.session_state["回答内容_web"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if '回答次数_web' not in st.session_state:
    st.session_state['回答次数_web'] = 1

if st.button('重新开始一个回答'):
    del st.session_state["回答内容_web"]
    del st.session_state["messages_web"]
    del st.session_state["回答次数_web"]
    st.session_state["回答内容_web"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state["messages_web"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state['回答次数_web'] = 1
    # 清空文本输入框的内容
    user_input = ""

for msg in st.session_state["messages_web"]:
    st.chat_message(msg["role"]).write(msg["content"])


@tool("search_database", return_direct=False)
def search_database(query: str) -> str:
    """当问题中涉及到南山的工作要求以及流程，用这个工具很有用"""
    query_message = ask_page.query_message_langchain(query=query, token_budget=2000, )
    return query_message


search = DuckDuckGoSearchRun()


def search_web(query: str) -> str:
    # query_message = search(f'site:nanshan.edu.cn {query}')
    query_message = search(query)
    return query_message


tools = [
    Tool(
        name="search_database",
        func=search_database,
        description="当问题中涉及到南山的工作要求以及流程，用这个工具很有用",
    ),
    Tool(
        name="search_web",
        func=search_web,
        description="当问题中没有提到南山学院的规章制度，使用这个工具",
    ),
]





from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish




first_template = """严格按照下面的格式回答有关南山学院的问题,输入一个问题，你来决定和思考使用哪个操作。

请严格按照以下格式回答:


问题:(您需要回答的输入的问题)
思考:(您应该考虑要做什么)
操作:(search_database or search_web)
操作输入:(输入的关键词)

使用上述模板，您可以按照以下方式填充每个部分：

问题: 烟台南山学院是一所什么类型的学校？
思考: 我应该搜索相关的数据库或者网页来获取关于烟台南山学院的信息。
操作: search_web
操作输入: 烟台南山学院类型


{question}


"""

template = """结合下面的文章来回答问题，使用中文回答。


{history}

{question}?

搜索结果：
{context}

"""


llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv('OPENAI_API_KEY'), streaming=True,temperature=0.2)



first_prompt = PromptTemplate(
    input_variables=['question'],
    template=first_template
)

prompt = PromptTemplate(
    input_variables=["context",'question','history'],
    template=template,
)
first_chain=LLMChain(llm=llm, prompt=first_prompt)
llm_chain = LLMChain(llm=llm, prompt=prompt,verbose=True)
def output(llm_output: str):
    regex = r"操作:(.*?)[\n]*操作输入:[\s]*(.*)"
    match = re.search(regex, llm_output, re.DOTALL)
    if not match:
        return AgentFinish(
            return_values={"output": llm_output},
            log=llm_output,
        )
       # raise ValueError(f"无法解析LLM输出:`{llm_output}`")
    action = match.group(1).strip()
    action_input = match.group(2)
    # Return the action and action input
    return [AgentAction(
        tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
    )]

def result_format(response: str):
    pattern = r"最终答案: (.*?)(?:\n|$)"
    matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
    if matches:
        final_answer = matches[0]
        final_answer_parts = re.split(r"。|；", final_answer)
        final_answer_parts = [part.strip() for part in final_answer_parts if part.strip()]
        return final_answer_parts
    else:
        return []


def tool_result(query):
    intermediate_steps=[]
    kwargs=[]
    tools=[]
    result=[]
    placeholder = st.empty()

    while len(intermediate_steps)<3:
        if len(intermediate_steps) == 0:
            placeholder.caption('正在思考....')
            d = first_chain.run(query)
            s = output(d)
            st.caption(d)
            intermediate_steps.append(s)
            kwargs.append(s[0].tool_input)
            tools.append(s[0].tool)
        if len(intermediate_steps) == 1:
            placeholder.empty()
            placeholder.caption(f'正在{tools[0]}::sunglasses:....')
            intermediate_steps.append(AgentAction(tool=tools[0], tool_input=kwargs[0], log=""))
            result.append(tool_result_(tools[0], kwargs[0]))
        elif len(intermediate_steps) == 2 and tools[0]=='search_database':
            placeholder.empty()
            placeholder.caption('搜索完成')
            placeholder.caption('正在查找网上数据[search_web]:sunglasses:......')
            intermediate_steps.append(AgentAction(tool="search_web", tool_input=kwargs[0], log=""))
            result.append(tool_result_("search_web", kwargs[0]))
            placeholder.empty()
        elif len(intermediate_steps) == 2 and tools[0]=='search_web':
            placeholder.empty()
            placeholder.caption('搜索完成')
            placeholder.caption('正在查找数据库[search_database]:sunglasses:......')
            intermediate_steps.append(AgentAction(tool="search_database", tool_input=kwargs[0], log=""))
            result.append(tool_result_("search_database", kwargs[0]))
            placeholder.empty()
    return result

def tool_result_(tool,tool_input):
    for tool_up in tools:
        if tool_up.name==tool:
            result=tool_up.run(tool_input)
            return result






if prompt := st.chat_input(placeholder="在这打字，回答问题"):
    st.session_state['messages_web'].append({"role": "user", "content": prompt})
    st.session_state["回答内容_web"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        # st.container()
        # st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        history= "\n历史对话:\n" + "\n".join([f"{d['role']}: {d['content']}" for d in st.session_state["回答内容_web"][:-1]])
        result = tool_result(prompt)
        placeholder = st.empty()
        placeholder.caption('搜索完成🎉')
        placeholder.caption('我现在在思考，别催🎈🎈')

        response = llm_chain.run(({'context':result,'question':f'\n问题：{prompt}','history':history}))


        st.session_state["回答内容_web"].append({"role": "assistant", "content": response})
        st.session_state['messages_web'].append({"role": "assistant", "content": response})
        st.session_state['回答次数_web'] = st.session_state['回答次数_web'] + 1
        placeholder.empty()
        placeholder.caption('完成🎉')
        st.write(response)




    conversation_string = ""
    short_state_num = len(st.session_state["回答内容_web"])

    start_round = int(short_state_num * 3 / 10)
    end_round = int(short_state_num * 7 / 10)
    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_web"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num = len(conversation_string)
    if conversation_string_num > 2000 or st.session_state['回答次数_web'] > 4:
        del st.session_state["回答内容_web"][start_round: end_round]
        st.session_state['回答次数_web'] = 1
#
