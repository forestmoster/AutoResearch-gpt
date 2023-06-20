
import streamlit as st
import split
from docx import Document
import pinecrone_chunk

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )


uploaded_files = st.file_uploader("选择一个纯文本docx文件", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None and st.button('上传'):
        doc = Document(uploaded_file)
        # 对文档进行处理，例如读取内容、修改样式等
        # 例如，打印文档中的段落内容
        pinecrone_chunk.chunk(20, 600, 50,uploaded_file=uploaded_file)
        st.write('上传成功')








