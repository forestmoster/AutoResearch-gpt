import ast
import asyncio
import jieba.analyse
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.vectorstores import Chroma
import PyPDF2
import tempfile
import split
import requests
import json

import os

class WebChroma:
    def __init__(self, vector_folder):
        self.vector_folder = vector_folder
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.api_key=os.getenv('coreapikey')

    @staticmethod
    def download_pdf(url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
        }
        try:
            response = requests.get(url,headers=headers)
            if response.status_code == 200:
                return response.content
            else:
                return None
        except :
            return None

    @staticmethod
    def read_pdf(pdf_content):
        # 创建一个临时文件，并将PDF内容写入其中
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(temp_file.name)
        # 删除临时文件
        os.remove(temp_file.name)
        return pdf_reader

    @staticmethod
    def load_all_stopwords(dir_path):
        """
        Load all stopwords from all files in a directory and set them in jieba.
        """
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                jieba.analyse.set_stop_words(file_path)

    def pdf_text(self,url:str):
        pdf_content = self.download_pdf(url)
        if pdf_content:
            # 创建PDF阅读器对象
            pdf_reader = self.read_pdf(pdf_content)
            # 逐页读取文本并存储在一个字符串中
            all_text = ""
            for page in pdf_reader.pages:
                all_text += page.extract_text()
            # 打印PDF文档的所有内容
            return all_text
        else:
            return("Failed to download the PDF.")


    def search_doi(self,dois:str):
        if isinstance(dois, list):
            my_list = dois
        elif isinstance(dois, str):
            if dois.startswith("[") and dois.endswith("]"):
                try:
                    my_list = ast.literal_eval(dois)
                except (ValueError, SyntaxError):
                    st.write("提供的 URL 列表格式不正确。")
                    return
            else:
                # 假设它是一个单一的 URL
                my_list = [dois]
        else:
            st.write("不支持的doi类型。")
            return
        down_url=[]
        for doi in my_list:
            search_params = {
                              "doi": doi
                                }
            url = f'https://api.core.ac.uk/v3/discover'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer {}'.format(self.api_key)  # 替换为您的API密钥
            }
            response = requests.post(url, data=json.dumps(search_params), headers=headers)
            results = response.json()
            if response.status_code == 200:
                down_url.append(results['fullTextLink'])
            else:
                continue
        return down_url

    def search_research_title_url_abstract(self,q: str, entity_type: str = 'works', limit: int = 10):
        search_params = {
            "q": q,
            "limit": 50,
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
            'Authorization': 'Bearer {}'.format(self.api_key)  # 替换为您的API密钥
        }
        response = requests.post(url, data=json.dumps(search_params), headers=headers)
        results = response.json()

        # scrollId={results['scrollId']}

        if results['results'] and response.status_code == 200:
            out_results = []
            for result in results['results']:
                if result['links'][0]['type'] == 'download':
                    year = result['yearPublished']
                    authors = result['authors']
                    title = result['title']
                    link = result['links'][0]
                    abstract = result['abstract']
                    s = {'year': year, 'authors': authors, 'title': title, 'url': link, 'abstract': abstract}
                    out_results.append(s)
            return out_results[:limit]
        else:
            return None

    def process_pdf(self, url, title='', authors='', year='', abstract='',MAX_TOKENS:int=1000):
        strings = []
        self.load_all_stopwords('./stopwords-master')
        title_year_authors = f'year:{year}，title:{title}，authors:{authors}'
        if abstract is None or not isinstance(abstract, str):
            abstract = ''
        tags = jieba.analyse.extract_tags(abstract, topK=10)
        st.caption((title, url))
        pdf_content = self.download_pdf(url)
        if pdf_content is None:
            return
        try:
            pdf_reader = self.read_pdf(pdf_content)
        except:
            return
        try:
            for page in pdf_reader.pages:
                all_text = page.extract_text().replace('...', '').replace('..', '').strip()
                formatted_url = f'url:{url}'
                formatted_tags = f'abstract keyword:{tags}'
                text = [all_text]
                strings.append((title_year_authors, formatted_url, formatted_tags, text))
        except UnicodeDecodeError:
            return
        MAX_TOKENS = MAX_TOKENS  # Adjust as needed
        wikipedia_strings = []
        for section in strings:
            wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))

        ids = [f"{url}_{i}" for i in range(1, len(wikipedia_strings) + 1)]
        docsearch = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                      persist_directory=self.vector_folder, ids=ids)
        return docsearch

    from concurrent.futures import ThreadPoolExecutor
    async def process_pdf_async(self, url, title='', authors='', year='', abstract='',MAX_TOKENS:int=1000):
        strings = []
        self.load_all_stopwords('./stopwords-master')
        title_year_authors = f'year:{year}，title:{title}，authors:{authors}'
        if abstract is None or not isinstance(abstract, str):
            abstract = ''
        tags = jieba.analyse.extract_tags(abstract, topK=10)
        st.caption((title, url))
        pdf_content = self.download_pdf(url)
        if pdf_content is None:
            return
        try:
            pdf_reader = self.read_pdf(pdf_content)
        except:
            return
        try:
            for page in pdf_reader.pages:
                all_text = page.extract_text().replace('...', '').replace('..', '').strip()
                formatted_url = f'url:{url}'
                formatted_tags = f'abstract keyword:{tags}'
                text = [all_text]
                strings.append((title_year_authors, formatted_url, formatted_tags, text))
        except UnicodeDecodeError:
            return
        MAX_TOKENS = MAX_TOKENS  # Adjust as needed
        wikipedia_strings = []
        for section in strings:
            wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))

        ids = [f"{url}_{i}" for i in range(1, len(wikipedia_strings) + 1)]
        Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                      persist_directory=self.vector_folder, ids=ids)


    def search_research_articles(self,query: str,search_query:str):
        st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
        results = self.search_research_title_url_abstract(query)
        if results is None:
            return '没有找到论文，你可以去其他网址下载pdf后，然后进行分析'
        for result in results:
            strings = []
            url = result['url']['url']
            title = result['title']
            abstract = result['abstract']
            authors = result['authors']
            year = result['year']
            docsearch=self.process_pdf(url,title,abstract,authors,year,2000)
        docs = docsearch.similarity_search(search_query, k=7)
        return docs

    async def search_research_articles_async(self,query: str):
        st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
        results = self.search_research_title_url_abstract(query)
        if results is None:
            return '没有找到论文，你可以去其他网址下载pdf后，然后进行分析'
        tasks = []
        for result in results:
            strings = []
            url = result['url']['url']
            title = result['title']
            abstract = result['abstract']
            authors = result['authors']
            year = result['year']
            task = self.process_pdf_async(url, title, abstract, authors, year, 2000)
            tasks.append(task)
        docsearches = await asyncio.gather(*tasks)
        return docsearches

    # async def read_research_articles(self,input_str: str):
    #     st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
    #     my_list = ast.literal_eval(input_str)
    #     tasks = []
    #     for url in my_list:
    #         url = url
    #         title = ''
    #         abstract = ''
    #         authors = ''
    #         year = ''
    #         task = self.process_pdf_async(url, title, abstract, authors, year, 1000)
    #         tasks.append(task)
    #     docsearches = await asyncio.gather(*tasks)
    #
    #     # docs = docsearches.similarity_search(input_str, k=10)
    #     return docsearches
    def read_research_articles(self,input_url: str,search_query:str=''):
        # global docsearch
        st.write('正在下载论文，这个过程可能持续几分钟，请耐心等待。。。。')
        if isinstance(input_url, list):
            my_list = input_url
        elif isinstance(input_url, str):
            if input_url.startswith("[") and input_url.endswith("]"):
                try:
                    my_list = ast.literal_eval(input_url)
                except (ValueError, SyntaxError):
                    st.write("提供的 URL 列表格式不正确。")
                    return
            else:
                # 假设它是一个单一的 URL
                my_list = [input_url]
        else:
            st.write("不支持的 URL 类型。")
            return
        # my_list=input_url
        strings = []
        for url in my_list:
            url = url
            title = ''
            abstract = ''
            authors = ''
            year = ''
            docsearch = self.process_pdf(url, title, abstract, authors, year, 1000)
        if search_query=='':
            search_query='content'
        docs = docsearch.similarity_search(search_query, k=7)
        return docs

    def search_Cache(self,query: str):
        wikipedia_strings = []
        db3 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                persist_directory=self.vector_folder)
        answer = db3.similarity_search(query, k=3)
        return answer

    def get_ids(self):
        wikipedia_strings = []
        db3 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                persist_directory=self.vector_folder)
        return db3.get()['ids']

    def delete_vector_database(self, title):
        title = title[0]
        wikipedia_strings = []
        db4 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                persist_directory=self.vector_folder)
        all_ids = self.get_ids()
        ids_to_delete = [id for id in all_ids if id.startswith(f"{title}_")]
        for id in ids_to_delete:
            db4.delete(id)

    def delete_all_vector_database(self):
        wikipedia_strings = []
        db4 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union",
                                persist_directory=self.vector_folder)
        all_ids = self.get_ids()
        db4.delete(all_ids)

def extract_titles_from_ids(ids):
    # 使用set来确保title是唯一的
    titles = set()
    for id in ids:
        title = id.split("_")[0]  # 假设id的格式是"filename_number"
        titles.add(title)
    return list(titles)

def streamlit_sidebar_delete_database(WebChroma):
    all_ids = WebChroma.get_ids()
    titles = extract_titles_from_ids(all_ids)
    option = st.sidebar.multiselect('选择数据库中的文件并且删除', titles)
    button = st.sidebar.button(label='删除')
    if option and button:
        WebChroma.delete_vector_database(option)
        st.sidebar.success(f'你已经成功删除{option}', icon="✅")
    st.sidebar.caption(f'数据库中的文件:{titles}', )

#
if __name__ == "__main__":
    chroma=WebChroma('C:/Users/Administrator.DESKTOP-TO9EK6D/Desktop/tmp')
#     s=asyncio.run(chroma.search_research_articles_async('ai'))
#     s=chroma.search_research_articles('ai')
#     s=chroma.search_Cache('ai')
    s=chroma.read_research_articles("['https://core.ac.uk/download/20659569.pdf']",'ai')
    print(s)