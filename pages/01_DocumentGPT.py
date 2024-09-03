from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
import streamlit as st

st.set_page_config(page_title="DocumentGPT", page_icon="📃")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


# 사용자가 올린 파일이 변경되지 않았다고 한다면, 이 함수는 실행되지 않는다.
# 그 대신에 건너뛰고, 실행하는 대신에 기존에 반환했던 값을 다시 반환한다.
# st.cache는 deprecate되어서 cache_data로 변경
# 실행된 상태에서 캐시를 삭제해도, 캐시가 히트되지 않고, 메모리상의 응답을 반환한다.
# 즉, 서비스를 재시작하기 전에는 캐시파일이 생성되지 않는다.(메모리삭제 타이밍은 확인 필요함)
# input을 streamlit이 hashing해서 관리하고 있다고 함.(확인필요)
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")
st.markdown(
    """
Welcome!
 
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
    """
)

# 좌측에서 X를 눌러서 파일을 삭제하면, 아래에 file=False가 되므로 메시지가 사라진다.
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt, .pdf or .docx file",
        type=["txt", "pdf", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file....")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        response = chain.invoke(message)
        send_message(response.content, "ai")

        # 바로 위의 chain과 동일한 의미를 갖는다.
        # docs = retriever.invoke(message)
        # docs = "\n\n".join(document.page_content for document in docs)
        # prompt = prompt.format_messages(context=docs, question=message)
        # llm.predict_messages(prompt)


else:
    # 사이드바에서 파일삭제시에 메모리에 있는 messages를 모두 초기화 해준다.
    st.session_state["messages"] = []
