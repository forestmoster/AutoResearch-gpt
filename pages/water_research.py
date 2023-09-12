import io
import os
import shutil

import pandas as pd
from docx.shared import Inches
from langchain import OpenAI, LLMChain
from langchain.agents import AgentExecutor, initialize_agent, AgentType, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import BaseLanguageModel, AgentAction, AgentFinish
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import streamlit as st
import uuid
from docx import Document
from langchain.tools import Tool
from typing import List, Union
from PDF_Chroma import PDFChroma, streamlit_sidebar_delete_database
from control_docx import initialize_doc_with_titles, extract_content_from_doc, delete_section_content, \
    add_or_update_section, add_or_update_tables, add_images_to_section
import re


# 创建tmp临时文件夹
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # 创建一个唯一的UUID
session_folder = os.path.join('tmp', st.session_state.session_id)
vector_folder = os.path.join(session_folder, 'vector')
if not os.path.exists(vector_folder):
    os.makedirs(vector_folder)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# 实例化pdfchromaxxxxxxxxxxxxxxxxxxxxx
PDFS=PDFChroma(vector_folder)
# 上传文件前端xxxxxxxxxxxxxxxxxxxxxxxxxxxx
st.title("💬 WATER RESEARCH ARICTICAL")
st.caption('你可以上传pdf文献和csv，来水一个论文，注意！！！最好5个5个上传pdf，csv文件不能取消')
uploaded_file = st.file_uploader("选择一个纯文本docx文件或者pdf文件",accept_multiple_files=True,label_visibility="hidden")
s=st.button(label='提交pdf到数据库')
if s:
    st.caption('稍等这个过程可能要几min')
    PDFS.upload_pdfs_chroma(uploaded_file)
# 在侧边栏中添加一个选择框xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
fruit = st.sidebar.selectbox(
    "Which do you want to write?",
    options=[
        "Title(标题) and Abstract (摘要)and Keywords (关键词)",
        "Introduction (引言)",
        "Methods(方法)and Results(结果)",
        "Discussion (讨论)",
        "Conclusion (结论)"
    ]
)


def list_directory_contents(directory_path):
    return os.listdir(directory_path)

def on_file_change(file):
    # 在这里处理文件上传后的操作
    return f'文件名: {file.name},文件大小: {file.size} bytes'
def csv_agent(
        llm: BaseLanguageModel,
        uploaded_files:uploaded_file,
) -> AgentExecutor:
    # 检查 uploaded_files 中的 CSV 文件数量
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    if len(csv_files) != 1:
        raise ValueError("请确保上传了一个且只有一个 CSV 文件!")
    df = pd.read_csv(csv_files[0])  # 使用第一个（也是唯一的）CSV文件
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, return_intermediate_steps=True)


# 在streamlit缓存中更新缓存，注意不是累加,而是更新
def update_session_cache(title, response_orgin, title_subfolder):
#xxxxxxxxxxxxxxx 将文字加入缓存xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    response = response_orgin['output']
    st.session_state.title_cache[title]["response"] = response
# xxxxxxxxxxxxxx 将表格添加进缓存xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if "intermediate_steps" in response_orgin:
        dataframes_list = []
        for observersion in response_orgin["intermediate_steps"]:
            if isinstance(observersion[1], pd.DataFrame):
                dataframes_list.append(observersion[1])
        st.session_state.title_cache[title]["dataframes"] = dataframes_list
# xxxxxxxxxxxxxxxx 将最新加入的图片添加进缓存xxxxxxxxxxxxxxxxxxxxxxxxx
    if os.path.exists(title_subfolder):
        all_image_files = [f for f in os.listdir(title_subfolder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # 如果title还未在st.session_state.title_cache中，则初始化它
        if title not in st.session_state.title_cache:
            st.session_state.title_cache[title] = {"images": []}
        # 根据时间排序获取最新的图片文件
        all_image_files.sort(key=lambda f: os.path.getmtime(os.path.join(title_subfolder, f)), reverse=True)
        # 计算新加入的图片数量
        new_added_count = len(all_image_files) - len(st.session_state.title_cache[title]["images"])
        # 仅获取最新上传的图片的路径
        new_image_paths_list = [os.path.join(title_subfolder, all_image_files[i]) for i in range(new_added_count)]
        # 更新session state
        st.session_state.title_cache[title]["images"] = new_image_paths_list

# xxxxxxxxxxxxxxxxxxxxxxxxx写一个chroma_db的agentxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def search(query):
    s=PDFS.search_upload_pdfs_chroma(query)
    return s
tools = [
    Tool(
        name = "Search",
        func=search,
        description="useful for when you need to answer questions about current events"
    )
]

template = """

   {answer_format}

   {question_guide}：{input}

   {background_information}


   """

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        if len(intermediate_steps) == 0:
            tools = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            tool_names = ",".join([tool.name for tool in self.tools])
            background_information = "\n"
            question_guide = ""
            answer_format = f"""Complete the objective as best you can.

   You can use the following tools:

   {tools}

   Please strictly follow the format below to answer:

   Question: (The question you need to answer)
   Thought: (What you should consider doing)
   Action: one of [{tool_names}]
   Action Input: (The keywords you input should be in English)

"""


        # 返回了背景信息
        elif 0 < len(intermediate_steps) < 2 :
            print(1)
            tools = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            tool_names = ",".join([tool.name for tool in self.tools])
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\nObservation:{observation}\nThought:"
            # Set the agent_scratchpad variable to that value
            background_information = thoughts
            question_guide ="Please develop your response by incorporating this background information. Ensure to specifically contrast your findings with relevant references, highlighting how your research differs from and improves upon these prior studies."
            answer_format = f'''"""Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
   '''
        else :
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_information
            # action, observation = intermediate_steps[0]
            thoughts = ""
            for action, observation in intermediate_steps:
                # thoughts += action.log
                thoughts += f"\nbackground_infomation:{observation}\n"
            # Set the agent_scratchpad variable to that value
            background_information = thoughts
            # background_information += f"{observation}\n"
            question_guide ="Please develop your response by incorporating this background information. Ensure to specifically contrast your findings with relevant references, highlighting how your research differs from and improves upon these prior studies."
            answer_format = ''
        kwargs["background_information"] = background_information
        kwargs["question_guide"] = question_guide
        kwargs["answer_format"] = answer_format
        print(intermediate_steps)
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str):
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        # if match == None:
        #     regex = r"操作:(.*?)[\n]*操作输入:[\s]*(.*)"
        #     match = re.search(regex, llm_output, re.DOTALL)
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # st.session_state["messages_prompt"].append(action_input.strip(" ").strip('"'))
        # Return the action and action input
        return [AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )]

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv('OPENAI_API_KEY'), streaming=True,
                 temperature=0.2)
output_parser = CustomOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)
search_database_agent = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# 初始化
# 创建缓存中的doc对象
if 'doc_methods' not in st.session_state:
    st.session_state.doc_methods =initialize_doc_with_titles()
doc = st.session_state.doc_methods
if "title" not in st.session_state:
    st.session_state["title"] = None
if "content_added" not in st.session_state:
    st.session_state["content_added"] = False
# 为每个 title 初始化缓存
if 'title_cache' not in st.session_state:
    st.session_state.title_cache = {
        "Title(标题) and Abstract (摘要)and Keywords (关键词)": {
            "response": "",
            "dataframes": [],
            "images": [],
        },
        "Introduction (引言)": {
            "response": "",
            "dataframes": [],
            "images": [],
        },
        "Methods(方法)and Results(结果)": {
            "response": "",
            "dataframes": [],
            "images": [],
        },
        "Discussion (讨论)": {
            "response": "",
            "dataframes": [],
            "images": [],
        },
        "Conclusion (结论)": {
            "response": "",
            "dataframes": [],
            "images": [],
        }
    }
# 根据用户的选择显示消息
# 文件名用随机数表示
st.caption(f"You selected: {fruit}")
random_file_name = str(uuid.uuid4())

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if fruit == "Title(标题) and Abstract (摘要)and Keywords (关键词)":
    if "prompt_received" not in st.session_state:
        st.session_state["prompt_received"] = False
    st.session_state["title"] = fruit
    title =fruit
    # 创建章节子文件夹
    title_subfolder = os.path.join(session_folder,title)
    if not os.path.exists(title_subfolder):
        os.makedirs(title_subfolder)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

    draft_content = extract_content_from_doc(doc)
    question = ''
    prompt = f'''
    question:{question},

    draft_content:{draft_content}，

    Craft a "Title, Abstract, and Keywords" section suitable for a scientific (SCI) paper with the following guidelines:

    1. **Title**: Create a concise and informative title that encapsulates the main thrust of the research. It should be clear, descriptive, and free from jargon. Ensure that it captures the essence of the study and piques the interest of the reader.

    2. **Abstract**: Compose a brief summary of the research that:
        a. Introduces the research topic and its relevance.
        b. Describes the main objectives and hypotheses of the study.
        c. Provides a brief overview of the methodology used.
        d. Highlights the main findings or results.
        e. Concludes with the broader implications or conclusions of the study.
       Ensure that the abstract is concise, usually within 200-300 words, and provides a clear snapshot of the entire paper.

    3. **Keywords**: List 4-6 keywords or key phrases that represent the core topics and concepts of the paper. These should be terms that are frequently used in the paper and central to the research topic. They will be used for indexing purposes and should be selected to maximize the paper's visibility in database searches.

    Ensure the tone is formal and the content is direct and to the point. Avoid using technical jargon in the title and abstract unless the target audience is domain-specific. The keywords should be relevant and commonly used in the field to improve searchability.
    '''

    if question := st.chat_input(placeholder="在这打字"):
        st.session_state["content_added"] = False
        response_orgin = search_database_agent(prompt, callbacks=[st_cb])
        update_session_cache(title, response_orgin, title_subfolder)
        st.session_state["prompt_received"] = True

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
elif fruit == "Introduction (引言)":
    if "prompt_received" not in st.session_state:
        st.session_state["prompt_received"] = False
    # s=extract_content_from_doc(doc)
    # st.write(s)
    st.session_state["title"] = fruit
    title =fruit
    # 创建章节子文件夹
    title_subfolder = os.path.join(session_folder,title)
    if not os.path.exists(title_subfolder):
        os.makedirs(title_subfolder)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    draft_content = extract_content_from_doc(doc)
    question = ''
    prompt = f'''question:{question},

        draft_content:{draft_content}，

        Write an introduction section suitable for a scientific (SCI) paper with the following considerations:

    1. **Introduction Paragraph**: Begin by introducing the broad background of the research field, presenting key statistics or widely accepted facts, and describing the main challenges or issues in the field.
    2. **Research Background**: Delve deeper into the specific background of the research question, giving an overview of the current research status and pointing out gaps or shortcomings in existing studies. Make sure to cite key literature to support these points.
    3. **Significance of the Research**: Explain the importance of the research topic, elaborate on the knowledge gap your research aims to fill, describe any practical problems it addresses, and if applicable, mention if the research introduces or validates new perspectives or theories.
    4. **Research Objectives and Hypotheses**: Clearly state the purpose or goals of the research, describe the main research question, and list any hypotheses or expected outcomes.
    5. **Scope and Limitations**: Define the boundaries of the research and detail any potential limitations or constraints, such as methodology, sample size, or biases.

    Ensure the tone is formal, and the content is supported by relevant scientific references. Use clear and coherent language, avoiding excessive technical jargon unless the target audience consists of domain experts. Keep the introduction concise, focusing solely on information directly relevant to the research question.
    '''

    if question := st.chat_input(placeholder="在这打字"):
        st.session_state["content_added"] = False
        response_orgin=search_database_agent(prompt, callbacks=[st_cb])
        update_session_cache(title, response_orgin, title_subfolder)
        st.session_state["prompt_received"] = True

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
elif fruit == "Methods(方法)and Results(结果)":
    st.session_state["title"]=fruit
    title = fruit
    st.session_state["title"]  = "Methods(方法)and Results(结果)"
    if "prompt_received" not in st.session_state:
        st.session_state["prompt_received"] = False
    # 创建章节子文件夹
    title_subfolder = os.path.join(session_folder,title)
    if not os.path.exists(title_subfolder):
        os.makedirs(title_subfolder)
    # 运行主程序
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    agent_wzm = csv_agent(llm=ChatOpenAI(temperature=0, model="gpt-4-0613"), uploaded_files=uploaded_file)
    # Extract the name of the first .csv file from the uploaded_file list
    csv_file_name = next((f.name for f in uploaded_file if f.name.lower().endswith('.csv')), None)
    draft_content = extract_content_from_doc(doc)
    # 运行agent
    if csv_file_name:
        if prompt := st.chat_input(placeholder="在这打字"):
            st.session_state["content_added"] = False
            response_orgin = agent_wzm(f'''question:{prompt},
            
                                draft_content:{draft_content},'''

                                       f'''Given a dataset from a specific research topic, please adhere to the following structure for your analysis and output:

            Methods:
            - Study Design: Describe the overall design of the study.
            - Study Subjects: Provide a detailed description of the study participants or subjects.
            - Data Collection: Describe how data was collected and any instruments or tools used.
            - Experimental Procedure: Detail any experimental methods and procedures.
            - Data Analysis: Explain the analytical methods used.

            Results:
            - Data Presentation: Present primary findings using tables, figures, or descriptive statistics. Highlight any significant patterns or results.
            - Statistical Analysis: Provide relevant test statistics, p-values, or confidence intervals.

            When generating or modifying plots related to the results:
            1. Use libraries like matplotlib or seaborn to create the plot.
            2. IMPORTANT: Save each generated image to the session's specific directory. Use the following exact command to save your plot:
                ```
                plt.savefig('./tmp/{st.session_state.session_id}/{title}/{random_file_name}.png'))
                ```
            3. Make sure all plots have clear titles, axis labels, and are easily interpretable.

            Ensure all generated content follows this format and includes both visual and tabular data.

            Execute the command 'Action:python_repl_ast' to generate the required output and save plots to the specified directory.
            ''', callbacks=[st_cb])
            #将文字,图片，表格添加进缓存
            update_session_cache(title, response_orgin, title_subfolder)
            st.session_state["prompt_received"] = True
    else:
        st.warning("Please upload a CSV file.")

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
elif fruit == "Discussion (讨论)":
    if "prompt_received" not in st.session_state:
        st.session_state["prompt_received"] = False
    st.session_state["title"] = fruit
    title = fruit
    # 创建章节子文件夹
    title_subfolder = os.path.join(session_folder,title)
    if not os.path.exists(title_subfolder):
        os.makedirs(title_subfolder)
    draft_content = extract_content_from_doc(doc)
    question=''
    prompt=f'''question:{question},
    
    draft_content:{draft_content}，
    
    Write a discussion section suitable for a scientific (SCI) paper with the following considerations:

1. Main findings: Begin by briefly revisiting the main objectives of the research and summarizing the key results.
2. Interpretation: Analyze and interpret the results, comparing them with existing literature. How do the findings align or contrast with previous research?
3. Unexpected results: Comment on any unexpected outcomes and hypothesize potential reasons.
4. Limitations: Acknowledge any limitations, such as methodology constraints, sample size, or biases.
5. Significance: Discuss the broader implications of the findings for the scientific community and potential real-world applications.
6. Future directions: Propose potential avenues for future research based on the current findings.
7. Conclusion: Conclude the discussion with a concise summary of the main points and their importance.

Ensure the tone is formal and the content is supported by relevant scientific references. Avoid speculative statements and maintain a neutral and objective perspective.
'''
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    if question := st.chat_input(placeholder="在这打字"):
        st.session_state["content_added"] = False
        response_orgin=search_database_agent(prompt, callbacks=[st_cb])
        update_session_cache(title, response_orgin, title_subfolder)
        st.session_state["prompt_received"] = True


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
elif fruit == "Conclusion (结论)":
    if "prompt_received" not in st.session_state:
        st.session_state["prompt_received"] = False
    st.session_state["title"] = fruit
    title = fruit
    # 创建章节子文件夹
    title_subfolder = os.path.join(session_folder,title)
    if not os.path.exists(title_subfolder):
        os.makedirs(title_subfolder)
    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
    draft_content = extract_content_from_doc(doc)
    question = ''
    prompt = f'''question:{question},

    draft_content:{draft_content}，

    Write a conclusion section suitable for a scientific (SCI) paper with the following guidelines:

    1. **Recapitulation**: Start by briefly restating the main objectives and key findings of the research.
    2. **Significance**: Emphasize the importance of the findings in terms of theoretical contributions, practical implications, and the advancement of knowledge in the field.
    3. **Recommendations**: If applicable, provide any actionable recommendations or suggestions based on the research findings. This might be particularly relevant for applied research.
    4. **Future Work**: Highlight potential directions for future research, either as a continuation of this study or as new related questions that emerged from the findings.
    5. **Final Thought**: End with a strong and memorable closing statement that reinforces the value and relevance of the study.

    Ensure that the tone remains formal. Avoid introducing new topics or questions not covered in the paper. The conclusion should provide clarity, closure, and encourage readers to reflect upon the research's importance.
    '''

    if question := st.chat_input(placeholder="在这打字"):
        st.session_state["content_added"] = False
        response_orgin=search_database_agent(prompt, callbacks=[st_cb])
        update_session_cache(title, response_orgin, title_subfolder)
        st.session_state["prompt_received"] = True


# xxxxxxxxxxxxxxxxxxx前端操作逻辑xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
title = st.session_state["title"]
st.write(st.session_state.title_cache[title]["response"])
for dataframe in st.session_state.title_cache[title]["dataframes"]:
    st.dataframe(dataframe)
for img_path in st.session_state.title_cache[title]["images"]:
    st.image(img_path)

if st.session_state["prompt_received"]:
    replace_or_add = st.button('Replace content in doc')
    title = st.session_state["title"]
    # 如果用户选择了替换内容
    if replace_or_add:
        # st.session_state["content_added"] = False  # 重置 content_added 的状态
        # 替换文档中的内容
        delete_section_content(doc, title)
        # 如果对应的title有response数据，那么更新文档的section
        current_title_cache = st.session_state.title_cache.get(title, {})
        # 如果对应的title有response数据，那么更新文档的section
        if "response" in current_title_cache:
            add_or_update_section(doc, title, current_title_cache["response"])
        # 如果对应的title有dataframes数据，那么更新文档的tables
        if "dataframes" in current_title_cache:
            add_or_update_tables(doc, title, current_title_cache["dataframes"])
        # 如果对应的title有images数据，那么更新文档的images
        if "images" in current_title_cache:
            add_images_to_section(doc, title, current_title_cache["images"])
        st.write("Content in doc has been replaced.")
        st.session_state["content_added"] = True
    # 如果用户没有选择替换内容
    elif not st.session_state["content_added"] and not replace_or_add:
        # 提取对应title的数据
        current_title_cache = st.session_state.title_cache.get(title, {})
        # 如果对应的title有response数据，那么更新文档的section
        if "response" in current_title_cache:
            add_or_update_section(doc, title, current_title_cache["response"])
        # 如果对应的title有dataframes数据，那么更新文档的tables
        if "dataframes" in current_title_cache:
            add_or_update_tables(doc, title, current_title_cache["dataframes"])
        # 如果对应的title有images数据，那么更新文档的images
        if "images" in current_title_cache:
            add_images_to_section(doc, title, current_title_cache["images"])
        st.session_state["content_added"] = True


byte_stream = io.BytesIO()
doc.save(byte_stream)
# 从字节流中读取数据
bytes_data = byte_stream.getvalue()
# 提供下载按钮
st.sidebar.download_button("点击此处下载分析word", bytes_data, "result.docx",
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
streamlit_sidebar_delete_database(PDFS)

#
# if os.path.exists(session_folder):
#     try:
#         shutil.rmtree(session_folder)
#         st.session_state["MR_image_path_list_cache"]=[]
#     except Exception as e:
#         st.write(f"Error removing folder {session_folder}: {e}")

