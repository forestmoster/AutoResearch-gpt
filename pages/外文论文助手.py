
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import ast
from typing import List
import re
import jieba.analyse
import openai
import pandas as pd
from langchain import LLMChain, OpenAI, PromptTemplate, text_splitter
from langchain.agents import AgentOutputParser, LLMSingleActionAgent, AgentExecutor, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentFinish, AgentAction
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import Tool
import streamlit as st
from langchain.vectorstores import Chroma
import PyPDF2
import tempfile
import os
import split
import requests
import json
api_key='8j9mqPr37oHsORDKJTWyeYMdBGgA5cZz'

def on_file_change(file):
    # 在这里处理文件上传后的操作
    return f'文件名: {file.name},文件大小: {file.size} bytes'
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

st.title("💬 外文文献助手")
st.caption('你可以联网查询和下载相关领域的外文文献，并撰写论文综述。你也可以上传一个pdf,对其进行分析')
uploaded_file = st.file_uploader("选择一个纯文本docx文件或者pdf文件",accept_multiple_files=False,label_visibility="hidden")
if uploaded_file is None:
    st.cache_resource.clear()
if "messages_article" not in st.session_state:
    st.session_state["messages_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if "回答内容_article" not in st.session_state:
    st.session_state["回答内容_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
if '回答次数_article' not in st.session_state:
    st.session_state['回答次数_article'] = 1
if "messages_wikipedia_strings" not in st.session_state:
    st.session_state["messages_wikipedia_strings"] = []
# if "messages_wikipedia_strings_read" not in st.session_state:
#     st.session_state["messages_wikipedia_strings_read"] = []
# if "messages_prompt" not in st.session_state:
#     st.session_state["messages_prompt"] = []
if "messages_wikipedia_缓存关键词" not in st.session_state:
    st.session_state["messages_wikipedia_缓存关键词"] = []

if st.button('重新开始一个回答'):
    del st.session_state["回答内容_article"]
    del st.session_state["messages_article"]
    del st.session_state["回答次数_article"]
    del st.session_state["messages_wikipedia_strings"]
    # del st.session_state["messages_wikipedia_strings_read"]
    # del st.session_state["messages_prompt"]
    del st.session_state["messages_wikipedia_缓存关键词"]
    st.session_state["回答内容_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state["messages_article"] = [{"role": "assistant", "content": "你好，同学，你想问什么？"}]
    st.session_state['回答次数_article'] = 1
    st.session_state["messages_wikipedia_strings"] = []
    # st.session_state["messages_wikipedia_strings_read"] = []
    # st.session_state["messages_prompt"] = []
    st.session_state["messages_wikipedia_缓存关键词"]=[]
    # 清空文本输入框的内容
    user_input = ""

for msg in st.session_state["messages_article"]:
    st.chat_message(msg["role"]).write(msg["content"])


def download_pdf(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except :
        return None

def read_pdf(pdf_content):
    # 创建一个临时文件，并将PDF内容写入其中
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_content)
        temp_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(temp_file.name)
    # 删除临时文件
    os.remove(temp_file.name)
    return pdf_reader

def pdf_text(url:str):
    pdf_content = download_pdf(url)
    if pdf_content:
        # 创建PDF阅读器对象
        pdf_reader = read_pdf(pdf_content)
        # 逐页读取文本并存储在一个字符串中
        all_text = ""
        for page in pdf_reader.pages:
            all_text += page.extract_text()
        # 打印PDF文档的所有内容
        return all_text
    else:
        return("Failed to download the PDF.")
# def search_doi(dois:str,key:str=api_key):
#     dois = ast.literal_eval(dois)
#     down_url=[]
#     for doi in dois:
#         search_params = {
#                           "doi": doi
#                             }
#         url = f'https://api.core.ac.uk/v3/discover'
#         headers = {
#             'Content-Type': 'application/json',
#             'Authorization': 'Bearer {}'.format(key)  # 替换为您的API密钥
#         }
#         response = requests.post(url, data=json.dumps(search_params), headers=headers)
#         results = response.json()
#         if response.status_code == 200:
#             down_url.append(results['fullTextLink'])
#         else:
#             continue
#     return down_url
@st.cache_resource
def read_upload_pdf(uploaded_file):
    strings = []
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
    except:
        raise ValueError("不支持的文件类型")
    all_text = ""
    for page in pdf_reader.pages:
        all_text += page.extract_text()
    title = uploaded_file.name
    url = ''
    tags = ''
    all_text = all_text.replace('...', '')
    all_text = all_text.replace('..', '')
    all_text = ' '.join(all_text.split())
    text = []
    text.append(all_text)
    strings.append((title, url, tags, text))
    wikipedia_strings = []
    MAX_TOKENS = 4000
    for section in strings:
        wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))
    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = Chroma.from_texts(wikipedia_strings, embeddings_model, collection_name="state-of-union")
    return docsearch
def search_read_upload_pdf(query:str):
    docsearch=read_upload_pdf(uploaded_file)
    answer=docsearch.similarity_search(query, k=3)
    return answer

def search_research_title_url_abstract(q:str,key:str=api_key,entity_type: str='works',limit:int=10):
    search_params = {
        "q": q,
        "limit":50,
        "scroll": True,
        "offset": 0,
        "scroll_id": "",
        "stats": True,
        "raw_stats": True,
        "exclude": [" "],
        "measure": True,
        "sort": ['_score:desc']
    }

    url = f'https://api.core.ac.uk/v3/search/{entity_type}'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(key)  # 替换为您的API密钥
    }
    response = requests.post(url, data=json.dumps(search_params), headers=headers)
    results=response.json()
    # scrollId={results['scrollId']}

    if response.status_code == 200:
        out_results=[]
        for result in results['results']:
            if result['links'][0]['type']=='download':
                year=result['yearPublished']
                authors=result['authors']
                title=result['title']
                link=result['links'][0]
                abstract= result['abstract']
                s = {'year': year, 'authors': authors, 'title': title, 'url': link, 'abstract': abstract}
                out_results.append(s)
        return out_results[:limit]
    else:
        return None

def search_research_articles(query:str):
    # if len(st.session_state["messages_wikipedia_strings"]) < 1:
        st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
        results=search_research_title_url_abstract(query)
        strings = []
        for result in results:
            url=result['url']['url']
            title=result['title']
            abstract=result['abstract']
            authors=result['authors']
            year=result['year']
            pdf_content = download_pdf(url)
            if pdf_content:
                # 创建PDF阅读器对象
                try:
                    pdf_reader = read_pdf(pdf_content)
                except :
                    continue
                # 逐页读取文本并存储在一个字符串中
                all_text = ""
                try:
                    for page in pdf_reader.pages:
                            all_text += page.extract_text()
                except UnicodeDecodeError:
                    # 在解码错误时跳过该页的文本提取
                    continue
                if abstract is None:
                    abstract = ''  # 将abstract的值更改为空字符串
                tags = jieba.analyse.extract_tags(abstract, topK=10)
                title = f'year:{year}，title:{title}，authors:{authors}'
                url=f'url:{url}'
                tags=f'abstract keyword:{tags}'
                all_text= all_text.replace('...', '')
                all_text = all_text.replace('..', '')
                all_text = ' '.join(all_text.split())
                text=[]
                text.append(all_text)
                strings.append((title,url, tags, text))
                st.caption((url,))
                st.session_state["messages_wikipedia_缓存关键词"].extend((title, url, tags))
            # 打印PDF文档的所有内容
            else:
                continue
        wikipedia_strings = []
        MAX_TOKENS = 1000
        for section in strings:
            wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))
        st.session_state["messages_wikipedia_strings"].extend(wikipedia_strings)
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(st.session_state["messages_wikipedia_strings"], embeddings)
        docs = docsearch.similarity_search(query, k=10)
        return docs
    # if len(st.session_state["messages_wikipedia_strings"]) > 1:
    #     embeddings = OpenAIEmbeddings()
    #     docsearch = Chroma.from_texts(st.session_state["messages_wikipedia_strings"], embeddings)
    #     docs = docsearch.similarity_search(st.session_state["messages_prompt"][-1], k=10)
    #     st.write(st.session_state["messages_prompt"][-1])
    #     return docs


def search_Cache(query:str):
    if len(st.session_state["messages_wikipedia_strings"]) > 1:
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(st.session_state["messages_wikipedia_strings"], embeddings)
        docs = docsearch.similarity_search(query, k=10)
        return docs
    if len(st.session_state["messages_wikipedia_strings"]) > 1:
        st.write('数据库没有东西')
def read_research_articles(input_str:str):
    # if len(st.session_state["messages_wikipedia_strings"]) < 1:
        st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
        my_list = ast.literal_eval(input_str)
        strings = []
        for url in my_list:
            url=url
            title=''
            abstract=''
            authors=''
            year=''
            pdf_content = download_pdf(url)
            if pdf_content:
                # 创建PDF阅读器对象
                try:
                    pdf_reader = read_pdf(pdf_content)
                except :
                    continue
                # 逐页读取文本并存储在一个字符串中
                all_text = ""
                try:
                    for page in pdf_reader.pages:
                            all_text += page.extract_text()
                except :
                    # 在解码错误时跳过该页的文本提取
                    continue
                if abstract is None:
                    abstract = ''  # 将abstract的值更改为空字符串
                tags = jieba.analyse.extract_tags(abstract, topK=10)
                title = f'year:{year}，title:{title}，authors:{authors}'
                url=f'url:{url}'
                tags=f'abstract keyword:{tags}'
                all_text= all_text.replace('...', '')
                all_text = all_text.replace('..', '')
                all_text = ' '.join(all_text.split())
                text=[]
                text.append(all_text)
                strings.append((title,url, tags, text))
                st.caption((title,url,))
                st.session_state["messages_wikipedia_缓存关键词"].extend((title, url, tags))

            # 打印PDF文档的所有内容
            else:
                continue
        wikipedia_strings = []
        MAX_TOKENS = 1000
        for section in strings:
            wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))
        st.session_state["messages_wikipedia_strings"].extend(wikipedia_strings)
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_texts(st.session_state["messages_wikipedia_strings"], embeddings)
        docs = docsearch.similarity_search(input_str, k=10)
        return docs
    # if len(st.session_state["messages_wikipedia_strings_read"]) > 1:
    #     embeddings = OpenAIEmbeddings()
    #     docsearch = Chroma.from_texts(st.session_state["messages_wikipedia_strings_read"], embeddings)
    #     docs = docsearch.similarity_search(st.session_state["messages_prompt"][-1], k=10)
    #     st.write(st.session_state["messages_prompt"][-1])
    #     return docs


tools = [
    Tool(
        name="search_research_title_url_abstract",
        func=search_research_title_url_abstract,
        description='''当只需要查找论文的标题或者查找论文的url，摘要，适合使用这个工具，
        If you use this tool,please use the action input format and the input must be automatically translate to English:(title:xxxxx OR abstract:xxxxx) AND (title:"xxxxxxx" OR abstract:"xxxxxxxx") AND year>20xx'''),
    Tool(
        name="search_research_articles",
        func=search_research_articles,
        description='''当只知道论文的关键词不知道url，用这个工具可以关键词搜索查看论文的全文,并且进行分析，If you use this tool,please use the action input format and the input must be automatically translate to English:(title:xxxxx OR abstract:xxxxx) AND (title:"xxxxxxx" OR abstract:"xxxxxxxx") AND year>20xx'''
    ),
    Tool(
        name="read_research_articles",
        func=read_research_articles,
        description='''用这个工具的前提是知道某几篇论文的url(url must have keyword:download)后，用这个工具可以查看论文的全文,并且进行分析，If you use this tool,please use the action input format and the input must be list:['https://core.ac.uk/download/xxxxxx.pdf',......]'''
    ),
    Tool(
        name="search_Cache",
        func=search_Cache,
        description='''用这个工具的前提是判断提问是否和缓存内容有关，用这个工具可以直接调取缓存中的数据，而不用下载论文'''
    ),
    # Tool(
    #     name="search_doi",
    #     func=search_doi,
    #     description='''当知道论文的doi时，用这个工具可以搜索到论文的url,If you use this tool,please use the action input format and the input must be list:['xxxxxxxx',......]:'''
    # ),
    Tool(
        name="search_read_upload_pdf",
        func=search_read_upload_pdf,
        description='''当您需要回答关于文件的问题时很有用。输入应为完整的问题'''
    ),
]



template = """

{answer_format}

历史对话记录:{history}

{cache}

{question_guide}：{input}

{background_infomation}


"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        search_present = any(step[0].tool in ['search_research_articles','read_research_articles','search_Cache'] for step in intermediate_steps)
        # background_infomation=[]
        # print(background_infomation)
        # 没有互联网查询信息
        if uploaded_file is not None and len(intermediate_steps)== 2:
            thoughts = ""
            for action, observation in intermediate_steps:
                # thoughts += action.log
                thoughts += f"\n背景信息:{observation}\n"
            # Set the agent_scratchpad variable to that value
            background_infomation = thoughts
            # background_infomation += f"{observation}\n"
            question_guide = "请结合这些背景信息回答我的问题，文章结尾需要有参考文献"
            history = st.session_state["回答内容_article"]
            answer_format = ''
        elif uploaded_file is not None:
            tools = "search_read_upload_pdf"
            tool_names = "search_read_upload_pdf"
            background_infomation = "\n"
            question_guide = ""
            history = st.session_state["回答内容_article"]
            answer_format =f'''像一个海盗一样进行回答
You can use the following tools:

{tools}

Please strictly follow the format below to answer:

问题:(The question you need to answer)
思考:(What you should consider doing)
操作:one of [{tool_names}]
操作输入:(The keywords you input should be English)
观察:(操作的结果)
... (这个 思考/操作/操作输入/观察 可以重复N次)
思考: I now know the final answer
最终答案: the final answer to the original input question
'''
        elif len(intermediate_steps)== 0:
            tools = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            tool_names=",".join([tool.name for tool in self.tools])
            background_infomation="\n"
            question_guide = ""
            history=st.session_state["回答内容_article"]
            # cache=f'缓存内容是和{}相关'
            answer_format = f"""Complete the objective as best you can.

You can use the following tools:

{tools}

Please strictly follow the format below to answer:

问题:(The question you need to answer)
思考:(What you should consider doing)
操作:one of [{tool_names}]
操作输入:(The keywords you input should be English)

上述模板可以按照这样填充：
问题:请帮我分析一下最新的关于xxxx的一篇文章，并进行分析。
思考:首先，我们需要找到最新的关于xxxx的文章。以便进行分析。
操作:one of [{tool_names}]
操作输入:(The keywords you input should be English)
"""

        # 返回了背景信息
        elif 0<len(intermediate_steps)<2 and search_present==False:
            tools = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
            tool_names = ",".join([tool.name for tool in self.tools])
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_infomation
            # background_infomation = "\n\n你还有这些已知信息作为参考：\n\n"
            # action, observation = intermediate_steps[0]
            thoughts = ""
            for action, observation in intermediate_steps:
                thoughts += action.log
                thoughts += f"\n观察:{observation}\n思考:"
            # Set the agent_scratchpad variable to that value
            background_infomation=thoughts
            # background_infomation += f"{observation}\n"
            question_guide = "请结合这些背景信息写出答案,最后需要有参考文献"
            history =st.session_state["回答内容_article"]
            answer_format =f'''像一个海盗一样进行回答
You can use the following tools:

{tools}

Please strictly follow the format below to answer:

问题:(The question you need to answer)
思考:(What you should consider doing)
操作:one of [{tool_names}]
操作输入:(The keywords you input should be English)
观察:(操作的结果)
... (这个 思考/操作/操作输入/观察 可以重复N次)
思考: I now know the final answer
最终答案: the final answer to the original input question

上述模板可以按照这样填充：

思考:阅读论文，获取关于xxxxx的研究综述
操作:xxxxx
操作输入:[{{title:xxxxx,abstract_keywords:xxxxxxx,url:https://core.ac.uk/download......}},......]
观察:(操作的结果)
思考:我找到答案了
最终答案:the final answer to the original input question
'''
        else :
            # 根据 intermediate_steps 中的 AgentAction 拼装 background_infomation
            # action, observation = intermediate_steps[0]
            thoughts = ""
            for action, observation in intermediate_steps:
                # thoughts += action.log
                thoughts += f"\n背景信息:{observation}\n"
            # Set the agent_scratchpad variable to that value
            background_infomation=thoughts
            # background_infomation += f"{observation}\n"
            question_guide = "请结合这些背景信息回答我的问题，文章结尾需要有参考文献"
            history =st.session_state["回答内容_article"]
            answer_format = ''

        kwargs["background_infomation"] = background_infomation
        kwargs["question_guide"] = question_guide
        kwargs["answer_format"] = answer_format
        kwargs["history"] = history
        kwargs["cache"] = f'缓存大概内容:{st.session_state["messages_wikipedia_缓存关键词"]}'
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"],
)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) :
        regex = r"操作：(.*?)[\n]*操作输入：[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match==None:
            regex = r"操作:(.*?)[\n]*操作输入:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
        if "最终答案:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("最终答案:")[-1].strip()},
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
#
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", openai_api_key=os.getenv('OPENAI_API_KEY'), streaming=True,temperature=0.2)

output_parser = CustomOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n观察:"],
    allowed_tools=tool_names,
)

agent_wzm = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )


# if uploaded_file is not None and st.button('上传'):



if prompt := st.chat_input(placeholder="在这打字，进行提问"):
    st.session_state['messages_article'].append({"role": "user", "content": prompt})
    st.session_state["回答内容_article"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_wzm.run(prompt,callbacks=[st_cb])
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
    if len(st.session_state["messages_wikipedia_缓存关键词"])>50:
        st.write('数据库超缓存了！！！,请重修开始一个回答')
