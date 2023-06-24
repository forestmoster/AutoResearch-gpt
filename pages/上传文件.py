
import streamlit as st
import pinecrone_chunk
# 导入松果数据库
import pinecone
# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key="b0e7c072-995c-4406-8c41-12238d626882",
    environment="us-west4-gcp"  # find next to API key in console
)
index = pinecone.Index('openai')

st.title('现有数据库词条数:{}'.format(index.describe_index_stats()['total_vector_count']))

uploaded_file = st.file_uploader("选择一个纯文本docx文件或者pdf文件", accept_multiple_files=False)

if uploaded_file is not None and st.button('上传'):
    try:
        pinecrone_chunk.chunk_pdf(20, 600, 32, uploaded_file=uploaded_file)
    except:
        pinecrone_chunk.chunk_docx(20, 600, 32, uploaded_file=uploaded_file)
    st.write('上传成功')










