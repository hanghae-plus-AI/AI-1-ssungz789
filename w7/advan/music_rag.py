import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAI의 GPT와 임베딩 모델
from langchain.chains import ConversationalRetrievalChain # 대화형 검색 체인
from langchain_community.document_loaders import WebBaseLoader # 웹 페이지 로더
from langchain.text_splitter import CharacterTextSplitter # 텍스트 분할
from langchain_community.vectorstores import FAISS # 벡터db
from langchain.prompts import PromptTemplate  # 프롬프트 템플릿
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title('음악 추천 AI 🎵')

YOUTUBE_MUSIC_URLS = [
    "https://music.youtube.com/playlist?list=RDCLAK5uy_n9Fbdw7e6ap-98_A-8JYBmPv64v-Uaq1g",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mdh8VQYQelR3oiqOVF6Hc7jtNgScMhGKE",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_n_9K9mus5yEqLgPqHYS4JOXb1r4kNCQ98",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_l_UXph_4_pZLZIy2gKVkTZOVetlWpIUKY",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_n0TxkLvtpM2pFrZ6AoEPiS8H-hQBdL9wY",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_kXvyk5pZ5d9s0ETlZnGT0sc7nGprX6gBw",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_lAh86fcCB8JtDz_RgwB4sX2qpVmj5IyPQ",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nvHhGVn_BXQYMh7vXfQRb_6N9BDHiZcGc", 
    "https://music.youtube.com/playlist?list=RDCLAK5uy_ly6s4irLuZAcjEDwJmqcA_UtSipMyGrqw",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_lBr9g6Bz_pY7_r-XEZ9u8vABWb8nEzxmw",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nOWVE8v1LcClXxEAIQ9GqR5dGGWqVkNwM",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mfut9V_o1n6O78oX1syryP6R-_iLDvIEM",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_keXj2T2V0Xz3_qVbA9LOrxiPYZtt-Q9BU",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nzsK5ei_yKXQBvdqbRW6RbPZ-1Rtrhg8Y",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mx7sr3Fmm_qCVZbGm4Uh4LC_pasUxgQgk",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mB8PpL7LxEBE9zsKqVu8bqi2PfxEXjYpk",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_n7J6BxZw6_csLe8VKCzw3uVtV6x3Qe5-c",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_n4Xv9_ZxhX7HqYwmJSs7g7ckGYWkNqSCE",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_miHmulDEgq6dY_x0d3EDvhafXLH4mMdHk",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mfPFg3jWNB6Gs3E6W6R8wHX9w5cq86klQ",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_l_q-uSO7YXmXqcks8KXeLQj8ofPrWVrGk",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mh-rYtN7cPtLPDXJGFEmXZwE8y7KYxPlE",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nnL8-66KqPFqq6Tr_38NBNSv1YgFw3_wQ",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_kfTYmLr_EIx3CvhRAjpR5MvPJVcXq8C-g",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_lDQ9z1fzQY-ic0M2JSAThUqpO3p0X8UxE",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mYJuW8xCj7zqgK8gZ-Or8LFpXquqN1Gxc",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nEzXsf-qVaVp7vL9szqNn2ssNwXgDnQwE", 
    "https://music.youtube.com/playlist?list=RDCLAK5uy_kJ7JnK4VkWMkjgJtjQq_V4_v2Th_NzV8g",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_lBP6z_98GKp2r3ZkEfcW5X4zFv6YZu2Pk",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_mw9x9qXvqLAkwxBXRzr1R8X8qKzJN8UyE",  
    "https://music.youtube.com/playlist?list=RDCLAK5uy_nz1Y3KA7VXZPjXLP3weKwYhQETOrX7pIY"   
]

# 시스템 프롬프트 정의 - gpt 도움 받음
template = """당신은 음악 추천 AI입니다.

Context: {context}
Question: {question}

사용자의 감정 상태에 따라 적절한 음악을 추천해주세요.
다음과 같은 형식으로 반드시 답변해주세요:

1. 추천 플레이리스트:
   * [노래제목] - [가수]
     - 링크: [YouTube Music 링크]
     - 추천 이유: [곡의 분위기와 추천 이유]

반드시 최소 5곡을 추천하고, 모든 곡에 대해 실제 YouTube Music 링크를 포함해야 합니다."""

# 프롬프트 객체를 생성하고 템플릿에서 사용할 변수를 설정
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 체인 초기화
@st.cache_resource
def initialize_chain():
   # 유튜브 링크 로드
   loader = WebBaseLoader(YOUTUBE_MUSIC_URLS)
   documents = loader.load()
   
   # 텍스트 분할
   text_splitter = CharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50
   )
   texts = text_splitter.split_documents(documents)
   
   # 임베딩 및 벡터Db 설정
   embeddings = OpenAIEmbeddings()
   vectorstore = FAISS.from_documents(texts, embeddings)
   
   # GPT 모델 설정
   llm = ChatOpenAI(
       model_name="gpt-4",
       temperature=0.7
   )
   
   # 체인 생성
   chain = ConversationalRetrievalChain.from_llm(
       llm=llm,
       retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # 상위 3개 문서 검색?? gpt가 알려줌
       combine_docs_chain_kwargs={"prompt": PROMPT},
       return_source_documents=True
   )
   
   return chain

chain = initialize_chain()

# 채팅 기록 저장
if "messages" not in st.session_state:
   st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 저장된 채팅 메시지들을 화면에 표시
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

# 유저 입력 받고 처리
if prompt := st.chat_input("현재 기분이나 상황을 알려주세요"):
  # 유저 메시지 저장
   st.session_state.messages.append({"role": "user", "content": prompt})
   
   # 유저 메시지 표시
   with st.chat_message("user"):
       st.markdown(prompt)
   
   # AI 대답 생성 / 표시
   with st.chat_message("assistant"):
      # 체인에 질문이랑 대화 기록 전달
       response = chain({"question": prompt,
                         "chat_history": st.session_state.chat_history})
       st.markdown(response["answer"])
       st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
       st.session_state.chat_history.extend([(prompt, response["answer"])])
