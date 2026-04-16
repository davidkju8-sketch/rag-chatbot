# app.py
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv("./data/.env ")
import streamlit as st

api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")


# =========================
# 2. 세션별 채팅 기록 저장소
# =========================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def trim_chat_history(session_id: str, max_messages: int = 4):
    """
    최근 max_messages개 발화만 남김
    """
    history = get_session_history(session_id)
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]


# =========================
# 3. PDF 처리
# =========================
@st.cache_resource
def process_pdf():
    # PDF 파일 경로를 네 환경에 맞게 수정
    pdf_path = "./2024_KB_부동산_보고서_최종.pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


# =========================
# 4. 벡터스토어 초기화
#    - 있으면 load
#    - 없으면 새로 만들고 save
# =========================
@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db_path = "faiss_db"

    if os.path.exists(db_path):
        vectorstore = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(db_path)

    return vectorstore


# =========================
# 5. 체인 초기화
# =========================
@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    template = """
당신은 KB 부동산 보고서 전문가입니다.
반드시 아래 검색된 문서 내용을 바탕으로만 답변하세요.
모르면 모른다고 답하세요.
가능하면 간단하고 이해하기 쉽게 설명하세요.

컨텍스트:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=api_key
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    base_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    chain = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chain


# =========================
# 6. Streamlit UI
# =========================
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇", page_icon="🏠")
    st.title("🏠 KB 부동산 보고서 AI 어드바이저")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 채팅 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("부동산 관련 질문을 입력하세요")

    if prompt:
        # 사용자 질문 표시
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # 체인 준비
        chain = initialize_chain()

        # 최근 4개 발화만 유지
        session_id = "streamlit_session"
        trim_chat_history(session_id, max_messages=4)

        # 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": session_id}}
                )
                st.markdown(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


if __name__ == "__main__":
    main()