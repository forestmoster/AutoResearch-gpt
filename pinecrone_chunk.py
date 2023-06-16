# imports
import os
import openai  # for calling the OpenAI API
from docx import Document
import split
import streamlit as st


def chunk (title_long:int,MAX_TOKENS:int,BATCH_SIZE:int,uploaded_file:str):
    # models
    openai.api_key = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = "text-embedding-ada-002"
    GPT_MODEL = "gpt-3.5-turbo"
    openai.Model.list()
    strings=[]
    text=[]
    doc = Document(uploaded_file)
    # 从文档中获取文件名和标题名字
    file_name = uploaded_file.name
    for paragraph in doc.paragraphs:
        # 获取段落内容
        text1 = paragraph.text.strip()
        # 获取段落标题
        if 0< len(text1) <= title_long:
            titles=text1
        # 构建包含标题的字符串
        if len(text1) > title_long:
            text1=text1
            text.append(text1)
        if len(text)>0:
            string = f"标题,{titles}:{text}"
        # 将文件名、标题和字符串添加到file_info列表中
            strings.append((file_name,titles,text))
            text = []

    # # 对文件进行切分
    wikipedia_strings = []
    MAX_TOKENS = MAX_TOKENS
    for section in strings:
        wikipedia_strings.extend(split.split_strings_from_subsection_word(section, max_tokens=MAX_TOKENS))
    st.write(f"{len(strings)}  sections split into {len(wikipedia_strings)} strings.")


# 导入松果数据库
    import pinecone
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key="b0e7c072-995c-4406-8c41-12238d626882",
        environment="us-west4-gcp"  # find next to API key in console
    )

    # check if 'openai' index already exists (only create index if not)
    if 'openai' not in pinecone.list_indexes():
        pinecone.create_index('openai', dimension=1536)
    # connect to index
    index = pinecone.Index('openai')

    # from datasets import load_dataset
    #
    # # load the first 1K rows of the TREC dataset
    # trec = load_dataset('trec', split='train[:500]')
    from tqdm.auto import tqdm  # this is our progress bar

    batch_size = BATCH_SIZE  # process everything in batches of 32
    for i in tqdm(range(0, len(wikipedia_strings), batch_size)):
        # set end position of batch
        i_end = min(i + batch_size, len(wikipedia_strings))
        # get batch of lines and IDs
        lines_batch = wikipedia_strings[i: i + batch_size]
        ids_batch = [str(n) for n in range(i, i_end)]
        # create embeddings
        res = openai.Embedding.create(input=lines_batch, engine=EMBEDDING_MODEL)
        embeds = [record['embedding'] for record in res['data']]
        # prep metadata and upsert batch
        meta = [{'text': line} for line in lines_batch]
        to_upsert = zip(ids_batch, embeds, meta)
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)


