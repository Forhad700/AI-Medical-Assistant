import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


load_dotenv()

st.set_page_config(page_title="🩺 Medical Assistant", page_icon="💊")
st.title("🩺👩🏻‍⚕️ AI Medical Assistant")
st.caption("Looking for Medical Answers? 💉 Let’s Talk! 💬")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ API not Found!. Try Again Later.")
    st.stop()


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return download_hugging_face_embeddings()


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index():
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local("index.faiss", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

@st.cache_resource(show_spinner="Loading LLM...")
def get_rag_chain(_retriever):
    chat_model = ChatGroq(model="llama-3.3-70b-versatile")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    rag_chain = create_retrieval_chain(_retriever, question_answer_chain)
    return rag_chain


embeddings = get_embeddings()
vectorstore = load_faiss_index()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain = get_rag_chain(retriever)


user_input = st.text_input("🩸 Ask Me a Question")


if user_input:
    with st.spinner("⏳ Generating Answer....."):
        try:
            result = rag_chain.invoke({"input": user_input})
            st.success(result["answer"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
