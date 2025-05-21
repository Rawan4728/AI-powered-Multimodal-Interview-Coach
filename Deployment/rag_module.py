import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

DB_DIR = "chroma_db (1)"

def extract_text_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)

def parse_cv(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_questions(docs):
    llm = ChatOpenAI(temperature=0)
    snippet = "\n\n".join(doc.page_content for doc in docs[:2])
    questions_prompt = (
        "Based on the candidate CV provided, generate:\n"
        "- 5 Technical questions (prefix with 'Technical:')\n"
        "- 5 Behavioral questions (prefix with 'Behavioral:')\n"
        f"\nCandidate CV Snippet:\n{snippet}"
    )
    return llm.predict(questions_prompt)

def answer_question(question, parsed_cv=None):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

    if parsed_cv:
        question += f"\n\nCandidate Background:\n{parsed_cv}"
    return qa_chain.run(question)
