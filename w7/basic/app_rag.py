import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import os

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 제목
st.title('스파르타 AI 챗봇 🤖')

# 초기 설정
@st.cache_resource
def initialize_chain():
    # 웹페이지 로드
    loader = WebBaseLoader("https://spartacodingclub.kr/blog/all-in-challenge_winner")
    documents = loader.load()

    # 텍스트 분할
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # 임베딩 및 벡터 데이터베이스 생성
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # ChatGPT 모델 설정
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Conversational Retrieval Chain 생성
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True
    )

    return chain

# 체인 초기화
chain = initialize_chain()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain(
            {"question": prompt, "chat_history": st.session_state.chat_history}
        )

        st.markdown(response["answer"])

        # 채팅 기록 업데이트
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.session_state.chat_history.extend([(prompt, response["answer"])])
