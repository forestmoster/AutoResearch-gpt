
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import LLMChain
from typing import List
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
if "回答内容_web"not in st.session_state:
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
    query_message =ask_page.query_message_langchain(query=query, token_budget=2000,)
    return query_message

search = DuckDuckGoSearchRun()



tools = [
    Tool(
        name="search_database",
        func=search_database,
        description="当问题中涉及到南山的工作要求以及流程，用这个工具很有用",
    ),
    Tool(
        name="search_web",
        func=search.run,
        description="当问题中没有提到南山学院的规章制度，使用这个工具",
    ),
]


# Set up the base template
template = """严格按照下面的格式回答问题，结合材料得出答案，最终答案字数要多，不能过滤，不能总结。

你可以使用以下工具:

{tools}

请严格按照以下格式回答:

问题:(您需要回答的输入的问题)
思考:(您应该考虑要做什么)
操作:(search_database or search_web)
操作输入:(输入的关键词)
观察:(操作的结果)
...（思考/操作/操作输入/观察可以重复N次）
思考:(我找到答案了)
最终答案:(原始输入问题的最终答案)


{input}
{agent_scratchpad}
"""







# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # 要使用的模板
    template: str
    # 可用工具的列表
    tools: List[Tool]
    def format(self, **kwargs) -> str:
        # 获取中间步骤（AgentAction、Observation 元组）
        # 以特定方式进行格式化
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n观察:{observation}\n思考:"
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ",".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input","intermediate_steps",],
)





class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) :
        # Check if agent should finish
        if "最终答案:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("最终答案:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
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


output_parser = CustomOutputParser()
#
from langchain.chat_models import ChatOpenAI
#
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'), streaming=True,temperature=0)
# # # LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n观察:"],
    allowed_tools=tool_names,
    max_iterations=2,
    return_intermediate_steps = True
)

agent_wzm = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )




if prompt := st.chat_input(placeholder="在这打字，回答问题"):
    st.session_state['messages_web'].append({"role": "user", "content": prompt})
    st.session_state["回答内容_web"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        history= "\n历史对话:\n" + "\n".join([f"{d['role']}: {d['content']}" for d in st.session_state["回答内容_web"][:-1]])

        response = agent_wzm.run(f'{history}'+f'\n问题：{prompt}', callbacks=[st_cb])
        st.session_state["回答内容_web"].append({"role": "assistant", "content": response})
        st.session_state['messages_web'].append({"role": "assistant", "content": response})
        st.session_state['回答次数_web'] = st.session_state['回答次数_web'] + 1
        st.write(response)




    conversation_string = ""
    short_state_num = len(st.session_state["回答内容_web"])

    start_round = int(short_state_num * 3 / 10)
    end_round = int(short_state_num * 7 / 10)
    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_web"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num = len(conversation_string)
    if conversation_string_num > 2000 or st.session_state['回答次数_web'] > 3:
        del st.session_state["回答内容_web"][start_round: end_round]
        st.session_state['回答次数_web'] = 1

