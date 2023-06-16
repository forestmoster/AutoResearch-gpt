# imports
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # for storing text and embeddings data
import ask_page
import os
import frozen_dir
import streamlit as st
# models
GPT_MODEL = "gpt-3.5-turbo"
# file_path= "./ytnanshanuniversity.csv"
# embeddings_path = frozen_dir.app_path()+file_path
# @st.cache_resource
# def read():
#     s = pd.read_csv(embeddings_path)
#     s['embedding'] = s['embedding'].apply(ast.literal_eval)
#     return s
# s=read()
# openai.api_key =input('输入api：')
# openai_api_key=openai.api_key
openai_api_key=os.getenv('OPENAI_API_KEY')

st.title('烟台南山学院ai助手')
st.caption('你可以查询有关南山学院的任何事情，你也可以自己上传:blue[docx文件]')

if '登录状态' not in st.session_state:
    st.session_state['登录状态'] = False
if '创建状态' not in st.session_state:
    st.session_state['创建状态'] = False
if '回答次数' not in st.session_state:
    st.session_state['回答次数'] = 1

def login_page():
    selection = st.empty()
    s = selection.container()
    s.title("登录")
    username = s.text_input("用户名")
    st.session_state['用户名'] = username
    password = s.text_input("密码", type="password")
    st.session_state['密码'] = password
    if s.button("登录"):
        if username == 'admin' and password == 'password':
            selection.empty()
            st.session_state['登录状态'] = True
            selection.empty()
    elif s.button("创建"):
        st.session_state['创建状态'] = True
        selection.empty()

def register_page():
    selection = st.empty()
    s = selection.container()
    s.title("注册")
    new_username = s.text_input("新用户名")
    new_password = s.text_input("新密码", type="password")
    if s.button("注册"):
        selection.empty()
        # 在这里进行注册逻辑，例如将新用户名和密码保存到数据库中
        st.success("注册成功！请登录账号。")
        st.session_state['创建状态'] =False
        login_page()
def query_page():
    if '回答内容' not in st.session_state:
        st.session_state['回答内容'] = []
    selection = st.empty()
    t = selection.container()

    response1=st.session_state['回答内容']
    count_pass=6-st.session_state['回答次数']
    response1_str = ''.join(response1)
    query = t.text_area('请输入500字以内提示语，最多连续提问{}次'.format(count_pass))
    query_response = query + response1_str
    if len(response1)>6:
        del st.session_state['回答内容']
    if t.button('重新开始一个回答,当前次数{}'.format(st.session_state['回答次数'])):
        del st.session_state['回答内容']
    if t.button("第{}次提交".format(st.session_state['回答次数'])):
        response = ask_page.ask(query=query_response, model=GPT_MODEL, token_budget=2000 - 500)
        st.write(response)
        st.write(':yellow[历史回答]',st.session_state['回答内容'])
        st.session_state['回答次数']=st.session_state['回答次数'] + 1
        response1.append(response)







# 显示登录或注册页面
if st.session_state['登录状态']==False and st.session_state['创建状态'] == False:
    login_page()
if  st.session_state['创建状态'] == True and st.session_state['登录状态']==False:
    register_page()
if st.session_state['登录状态']==True and st.session_state['回答次数'] <= 6:
    query_page()
if st.session_state['回答次数'] > 6:
    st.write('回答次数结束')



# count=0
# response1=[]
# while True:
#      count = count + 1
#      count_pass=6-count
#      if count <= 6:
#           if st.session_state['登录状态']==True:
#                response1_str = ''.join(response1)
#                query=st.text_area('请输入提示语，最多连续提问{}次'.format(count_pass))
#                query_response = query + response1_str
#                st.button("第{}次提交".format(count))
#                if st.button:
#                     response = ask.ask(query=query_response, df=s, model=GPT_MODEL, token_budget=2000 - 500)
#                     st.write('回答:', response)
#                     response1.append(response)
# count = 0
# response1 = []
#
# while True:
#      count += 1
#      count_pass = 6 - count
#      if count <= 6:
#           if st.session_state.get('登录状态', True):
#                response1_str = ''.join(response1)
#                query = st.text_area('请输入提示语，最多连续提问{}次'.format(count_pass,disabled=True))
#                query_response = query + response1_str
#                button_clicked = st.button("第{}次提交".format(count))
#                if button_clicked:
#                     response = ask.ask(query=query_response, df=s, model=GPT_MODEL, token_budget=2000 - 500)
#                     st.write('回答:', response)
#                     response1.append(response)

# count = 0
# response1 = []
#
# while True:
#     count += 1
#     count_pass = 6 - count
#
#     if count <= 6:
#         if st.session_state.get('登录状态', False):
#             response1_str = ''.join(response1)
#
#             query = st.text_area('请输入提示语，最多连续提问{}次'.format(count_pass), key='query_input', disabled=count > 1)
#             query_response = query + response1_str
#
#             if st.button("提交", key='submit_button', disabled=count > 1):
#                 response = ask.ask(query=query_response, df=s, model=GPT_MODEL, token_budget=2000 - 500)
#                 st.write('回答:', response)
#                 response1.append(response)


#
# count=0
# response1=[]
# while True:
#     count=count+1
#     if count<=6 :
#         response1_str = ''.join(response1)
#         query=input('请输入问题，最多连续提问6次，开始新对话输入N,退出输入Q,刷新数据库请输入Refresh：')
#         if query=='Q':
#             break
#         if query=='N':
#             response1 = []
#             count = 0
#             print('开始新的一轮对话')
#             continue
#         if query=='Refresh':
#             chunk.chunk(20,600,50,openai_api_key)
#             print('数据库更新成功')
#             break
#         query_response=query+response1_str
#         response = ask.ask(query=query_response,df=s,model=GPT_MODEL, token_budget=2000-500)
#         response1.append(response)
#         print(response)
#     else:
#         response1 = []
#         count = 0
#         print('对话达到次数，本轮对话结束')




