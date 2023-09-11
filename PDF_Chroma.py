import os
import uuid
import PyPDF2
import jieba.analyse
import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import split
# def load_all_stopwords(dir_path):
#     """
#     Load all stopwords from all files in a directory and set them in jieba.
#     """
#     for filename in os.listdir(dir_path):
#         if filename.endswith(".txt"):  # 假设所有的停用词文件都以.txt结尾
#             file_path = os.path.join(dir_path, filename)
#             jieba.analyse.set_stop_words(file_path)

# @st.cache_resource
# def upload_pdfs_chroma(uploaded_files, vector_folder,ids):
#     all_strings = []
#
#     for uploaded_file in uploaded_files:
#         # Skip the file if it's not a PDF
#         if not uploaded_file.name.lower().endswith('.pdf'):
#             continue
#         strings = []
#         try:
#             pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         except:
#             raise ValueError("不支持的文件类型")
#         all_text = ""
#         for page in pdf_reader.pages:
#             all_text += page.extract_text()
#         title = uploaded_file.name
#         url = ''
#         load_all_stopwords('./stopwords-master')
#         tags = jieba.analyse.extract_tags(all_text, topK=10)
#         tags_strings = " ".join(tags)
#         all_text = all_text.replace('...', '')
#         all_text = all_text.replace('..', '')
#         all_text = ' '.join(all_text.split())
#         text = []
#         text.append(all_text)
#         strings.append((title, url, tags_strings, text))
#         all_strings.extend(strings)
#     wikipedia_strings = []
#     MAX_TOKENS = 4000
#     for section in all_strings:
#         wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))
#     embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
#
#     docsearch = Chroma.from_texts(wikipedia_strings, embeddings_model, collection_name="state-of-union",
#                                   persist_directory=vector_folder,ids=ids)
#     return docsearch
#
#
# def search_read_upload_pdf(query:str,vector_folder,ids):
#     docsearch=upload_pdfs_chroma(uploaded_file,vector_folder,ids)
#     wikipedia_strings = []
#     db3 = Chroma.from_texts(wikipedia_strings, embedding_function, collection_name="state-of-union",
#                             persist_directory=vector_folder)
#     answer = db3.similarity_search(query,k=3)
#     return answer
#
# def get_ids(vector_folder):
#     wikipedia_strings = []
#     db3= Chroma.from_texts(wikipedia_strings, embedding_function, collection_name="state-of-union",
#                             persist_directory=vector_folder)
#     s=db3.get()
#     return s['ids']
#
# def delete_vector_database(vector_folder,ids):
#     wikipedia_strings = []
#     db4= Chroma.from_texts(wikipedia_strings, embedding_function, collection_name="state-of-union",
#                             persist_directory=vector_folder)
#     db4.delete(ids)

import os
import PyPDF2
from langchain.vectorstores import Chroma
import split
from langchain.embeddings import OpenAIEmbeddings
import jieba.analyse
import streamlit as st
class PDFChroma:

    def __init__(self, vector_folder):
        self.vector_folder = vector_folder
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

    @staticmethod
    def load_all_stopwords(dir_path):
        """
        Load all stopwords from all files in a directory and set them in jieba.
        """
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                jieba.analyse.set_stop_words(file_path)

    def upload_pdfs_chroma(self, uploaded_files):
        if uploaded_files:
            ids = []
            for file in uploaded_files:
                if file.name.lower().endswith('.pdf'):
                    ids.append(file.name)
            all_strings = []
            for uploaded_file in uploaded_files:
                if not uploaded_file.name.lower().endswith('.pdf'):
                    continue
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
                self.load_all_stopwords('./stopwords-master')
                tags = jieba.analyse.extract_tags(all_text, topK=10)
                tags_strings = " ".join(tags)
                all_text = all_text.replace('...', '').replace('..', '').strip()
                strings.append((title, url, tags_strings, [all_text]))
                all_strings.extend(strings)
            wikipedia_strings = []
            MAX_TOKENS = 2000
            for section in all_strings:
                wikipedia_strings.extend(split.split_strings_from_subsection_pdf(section, max_tokens=MAX_TOKENS))
            docsearch = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union", persist_directory=self.vector_folder, ids=ids)
            return docsearch

    def search_upload_pdfs_chroma(self, query):
        wikipedia_strings = []
        db3 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union", persist_directory=self.vector_folder)
        answer = db3.similarity_search(query, k=3)
        return answer


    def get_ids(self):
        wikipedia_strings = []
        db3 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union", persist_directory=self.vector_folder)
        return db3.get()['ids']

    def delete_vector_database(self, ids):
        wikipedia_strings = []
        db4 = Chroma.from_texts(wikipedia_strings, self.embedding_function, collection_name="state-of-union", persist_directory=self.vector_folder)
        db4.delete(ids)

def streamlit_sidebar_delete_database(PDFChroma):
    option = st.sidebar.multiselect('选择数据库中的文件并且删除', PDFChroma.get_ids(), )
    button = st.sidebar.button(label='删除')
    if option and button:
        PDFChroma.delete_vector_database(option)
        st.sidebar.success(f'你已经成功删除{option}', icon="✅")
    st.sidebar.caption(f'数据库中的文件:{PDFChroma.get_ids()}', )
#
# if 'session_id' not in st.session_state:
#     st.session_state.session_id = str(uuid.uuid4())  # 创建一个唯一的UUID
# session_folder = os.path.join('tmp', st.session_state.session_id)
# vector_folder = os.path.join(session_folder, 'vector')
# st.write(vector_folder)
# if not os.path.exists(vector_folder):
#     os.makedirs(vector_folder)
# #
# PDFChroma = PDFChroma(vector_folder)
# # #
# uploaded_file = st.file_uploader("选择一个纯文本docx文件或者pdf文件",accept_multiple_files=True,label_visibility="hidden")
# #
# if prompt := st.chat_input(placeholder="在这打字"):
# #
# #     ids=[]
# #     for file in uploaded_file:
# #         if file.name.lower().endswith('.pdf'):
# #             ids.append(file.name)
#     upload_pdfs_chroma(uploaded_file, )
# #     docs=PDFChroma.search_upload_pdfs_chroma(prompt)
# #     # PDFChroma.delete_vector_database('2')
# #     st.write('你好',docs)
# st.sidebar.caption(f'数据库中的文件:{PDFChroma.get_ids()}',)
# # streamlit_sidebar_delete_database(PDFChroma)
# # example_db._collection.delete(ids=[ids[-1]])
# st.write(PDFChroma.get_ids())





