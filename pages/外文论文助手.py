# if your deploy app in local you should not use it
import sys
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import shutil
import sys
import uuid
import pandas as pd
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import BaseLanguageModel
import streamlit as st
from Web_Chroma import WebChroma, streamlit_sidebar_delete_database
import os
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from FILE_Chroma import FileChroma

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # 创建一个唯一的UUID
session_folder = os.path.join('tmp', st.session_state.session_id)
vector_folder_web = os.path.join(session_folder, 'vector_web')
catch_picture = os.path.join(session_folder, 'catch_picture')
if not os.path.exists(vector_folder_web):
    os.makedirs(vector_folder_web)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)
if not os.path.exists(catch_picture):
    os.makedirs(catch_picture)
random_file_name = str(uuid.uuid4())
chromaweb = WebChroma(vector_folder_web)
chromfile = FileChroma(vector_folder_web)


def search_research_articles(query, search_query):
    return chromaweb.search_research_articles(query, search_query)


def read_research_articles(query, search_query: str):
    return chromaweb.read_research_articles(query, search_query)


def search_Cache(query: str):
    return chromaweb.search_Cache(query)


def search_research_title_url_abstract(query: str):
    return chromaweb.search_research_title_url_abstract(query)


def search_doi(dois: str):
    chromaweb.search_doi(dois)


@st.cache_resource
def read_upload_pdf(uploaded_files):
    return chromfile.upload_pdfs_chroma_catch(uploaded_files)


def search_read_upload_pdf(query: str):
    docsearch = read_upload_pdf(uploaded_file)
    answer = docsearch.similarity_search(query, k=3)
    return answer


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
# st.set_page_config(layout="wide")
st.title("💬 外文文献助手+丐版数据分析助手")
st.caption(
    '你可以联网查询相关领域的外文文献，并做初步的分析。你也可以上传一个pdf进行文本分析或者csv进行数据分析，绘制科研图纸')
uploaded_file = st.file_uploader("选择一个纯文本docx文件或者pdf文件", accept_multiple_files=False,
                                 label_visibility="hidden")
if uploaded_file:
    uploaded_file = [uploaded_file]
if uploaded_file is None:
    st.cache_resource.clear()
if "messages_article" not in st.session_state:
    st.session_state["messages_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if "回答内容_article" not in st.session_state:
    st.session_state["回答内容_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if '回答次数_article' not in st.session_state:
    st.session_state['回答次数_article'] = 1
if "messages_wikipedia_缓存关键词" not in st.session_state:
    st.session_state["messages_wikipedia_缓存关键词"] = []

if st.button('重新开始一个回答'):
    del st.session_state["回答内容_article"]
    del st.session_state["messages_article"]
    del st.session_state["回答次数_article"]
    del st.session_state["messages_wikipedia_缓存关键词"]
    st.session_state["回答内容_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state["messages_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state['回答次数_article'] = 1
    st.session_state["messages_wikipedia_缓存关键词"] = []
    chromaweb.delete_all_vector_database()
    # 清空文本输入框的内容
    user_input = ""

for msg in st.session_state["messages_article"]:
    st.chat_message(msg["role"]).write(msg["content"])


def csv_agent(
        llm: BaseLanguageModel,
        uploaded_files: uploaded_file,
) -> AgentExecutor:
    # 检查 uploaded_files 中的 CSV 文件数量
    csv_files = [f for f in uploaded_files if f.name.lower().endswith('.csv')]
    if len(csv_files) != 1:
        raise ValueError("请确保上传了一个且只有一个 CSV 文件!")
    df = pd.read_csv(csv_files[0])  # 使用第一个（也是唯一的）CSV文件
    return create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, return_intermediate_steps=True)


# xxxxxxxxxxxxxxxxxxxxxxxxxTOOLSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class search_research_title_url_abstract_input(BaseModel):
    """Inputs for get_current_stock_price"""
    query: str = Field(
        description='''the input must be automatically translate to English:(title:xxxxx OR abstract:xxxxx) AND (title:'xxxxxxx" OR abstract:"xxxxxxxx") AND year>20xx''')


class search_research_title_url_abstract_tool(BaseTool):
    name = "search_research_title_url_abstract"
    description = "当只需要查找论文的标题或者查找论文的url，摘要，适合使用这个工具"
    args_schema: Type[BaseModel] = search_research_title_url_abstract_input

    def _run(self, query: str):
        response = search_research_title_url_abstract(query)
        return response

    def _arun(self, query: str):
        raise NotImplementedError("search_research_title_url_abstract does not support async")


class read_research_articles_Input(BaseModel):
    query: str = Field(
        description='''You must provide URLs. Use the following format: ["https://core.ac.uk/download/xxxxxx.pdf", ...]      Note: The query must be a str.''')
    search_query: str = Field(
        escription="Enter the question you want to search for in research articles. Cannot be empty.")


class read_research_articles_Tool(BaseTool):
    name = "read_research_articles"
    description = "if you have urls ,you can use this to download and read articles"
    args_schema: Type[BaseModel] = read_research_articles_Input

    def _run(self, query: str, search_query: str):
        response = read_research_articles(query, search_query)
        return response

    def _arun(self, query: str, search_query: str):
        raise NotImplementedError("read_research_articles does not support async")


class search_doi_input(BaseModel):
    """Inputs for get_current_stock_price"""
    dois: str = Field(
        description='''You must provide dois. Use the following format: ["'xxxxxxxx',......]  Note: The query must be a list.''')


class search_doi_tool(BaseTool):
    name = "search_doi"
    description = "if you have doi ,you can use this to read articles"
    args_schema: Type[BaseModel] = search_doi_input

    def _run(self, dois: str):
        response = search_doi(dois)
        return response

    def _arun(self, query: str):
        raise NotImplementedError("search_doi does not support async")


class search_Cache_input(BaseModel):
    """Inputs for get_current_stock_price"""
    query: str = Field(description='''you need provide query to search content to answer question ''')


class search_Cache_tool(BaseTool):
    name = "search_Cache"
    description = "Use this tool to determine if your query is related to indexed content in the cache. Retrieve data directly from the cache without needing to download entire papers."
    args_schema: Type[BaseModel] = search_Cache_input

    def _run(self, query: str):
        response = search_Cache(query)
        return response

    def _arun(self, query: str):
        raise NotImplementedError("search_Cache does not support async")


class search_read_upload_pdf_input(BaseModel):
    """Inputs for get_current_stock_price"""
    query: str = Field(description='''you need provide a complete query to search content to answer question ''')


class search_read_upload_pdf_tool(BaseTool):
    name = "search_read_upload_pdf"
    description = "Use this tool to answer question with upload_pdf"
    args_schema: Type[BaseModel] = search_read_upload_pdf_input

    def _run(self, query: str):
        response = search_read_upload_pdf(query)
        return response

    def _arun(self, query: str):
        raise NotImplementedError("search_read_upload_pdf does not support async")


# xxxxxxxxxxxxxxxxxxxxxxxxxTOOLSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613", temperature=0)
tools = [search_research_title_url_abstract_tool(), read_research_articles_Tool(), search_doi_tool(),
         search_Cache_tool()]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
tools_pdf = [search_read_upload_pdf_tool()]
agent_pdf = initialize_agent(tools=tools_pdf, llm=llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

accepted_extensions = ('.csv',)
if prompt := st.chat_input(placeholder="在这打字，进行提问"):
    # 清除缓存图片clean_tmp
    if prompt == 'clean':
        work_directory = "./tmp"  # 请替换为您的工作目录路径
        # 遍历工作文件夹中的所有文件
        for filename in os.listdir(work_directory):
            file_path = os.path.join(work_directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 删除文件
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 删除目录
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        st.write("All file in the working directory have been deleted.")
        sys.exit()

    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        if uploaded_file:
            csv_file_name_csv = next((f.name for f in uploaded_file if f.name.lower().endswith('.csv')), None)
            csv_file_name_pdf = next((f.name for f in uploaded_file if f.name.lower().endswith('.pdf')), None)
            # if uploaded_file is not None and uploaded_file.name.lower().endswith(accepted_extensions):
            if csv_file_name_csv:
                agent_wzm = csv_agent(llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                                      uploaded_files=uploaded_file)
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response_orgin = agent_wzm(f'''question:{prompt},
                        history:{st.session_state['回答内容_article']},

            Whenever you're generating or modifying a plot, you must display it on Streamlit. The process involves the following steps:

            1. First, save the image to a temporary directory. You can use the following command: 'plt.savefig('./tmp/{st.session_state.session_id}/catch_picture/{random_file_name}.png')'.
            2. Second, ensure you have already imported Streamlit with 'import streamlit as st' at the beginning of your code.
            3. Third, after saving the image, you can display it on your Streamlit app using: 'st.image('./tmp/{st.session_state.session_id}/catch_picture/{random_file_name}.png')'.
            4. Finally, after it's been displayed, delete the image using: 'os.remove('./tmp/{st.session_state.session_id}/catch_picture/{random_file_name}.png')'.

            Remember, all plots should have a clear title and axis labels for better interpretation.

            you must use this 'Action:python_repl_ast' ''', callbacks=[st_cb])

                for observersion in response_orgin["intermediate_steps"]:
                    try:
                        if isinstance(observersion[1], pd.DataFrame):
                            st.dataframe(observersion[1])
                    except TypeError as e:
                        pass
                response = response_orgin['output']
            # 否则运行这个agent
            elif csv_file_name_pdf:
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = agent_pdf.run(f'''history:{st.session_state['回答内容_article']},
                question:{prompt},analyse the upload file ''', callbacks=[st_cb])
        else:
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(f'''history:{st.session_state['回答内容_article']},
            question:{prompt},''', callbacks=[st_cb])

        # 前端缓存处理
        st.session_state['messages_article'].append({"role": "user", "content": prompt})
        st.session_state["回答内容_article"].append({"role": "user", "content": prompt})
        st.session_state["回答内容_article"].append({"role": "assistant", "content": response})
        st.session_state['messages_article'].append({"role": "assistant", "content": response})
        st.session_state['回答次数_article'] = st.session_state['回答次数_article'] + 1
        st.write(response)

    conversation_string = ""
    short_state_num = len(st.session_state["回答内容_article"])
    start_round = int(short_state_num * 3 / 10)
    end_round = int(short_state_num * 7 / 10)
    for i in range(short_state_num):
        conversation_string += st.session_state["回答内容_article"][i]["content"] + "\n"
    # 调用计算文字的函数
    conversation_string_num = len(conversation_string)
    if conversation_string_num > 2000 or st.session_state['回答次数_article'] > 4:
        del st.session_state["回答内容_article"][start_round: end_round]
        st.session_state['回答次数_article'] = 1
    if len(st.session_state["messages_wikipedia_缓存关键词"]) > 50:
        st.write('数据库超缓存了！！！,请重修开始一个回答')

streamlit_sidebar_delete_database(chromaweb)