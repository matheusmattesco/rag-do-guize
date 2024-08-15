import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Diretório onde os PDFs estão armazenados
PDF_DIR = 'PDF'

def get_pdf_text(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as pdf:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Responda à pergunta o mais detalhadamente possível a partir do contexto fornecido, certifique-se de fornecer todos os detalhes. Se a resposta não estiver disponível
    no contexto, basta dizer "a resposta não está disponível no contexto". Não forneça uma resposta errada.\n\n
    Contexto:\n{context}\n
    Pergunta: \n{question}\n
    Resposta:\n
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def load_pdfs_and_create_index():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

    if not pdf_files:
        st.warning(f"Não foram encontrados arquivos PDF na pasta {PDF_DIR}.")
        return None

    all_text_chunks = []
    for pdf_file in pdf_files:
        file_path = os.path.join(PDF_DIR, pdf_file)
        raw_text = get_pdf_text(file_path)
        if raw_text:
            text_chunks = get_text_chunks(raw_text)
            all_text_chunks.extend(text_chunks)

    if all_text_chunks:
        return get_vector_store(all_text_chunks)
    return None

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    formatted_response = response["output_text"].replace("•", "\n•")
    
    st.markdown(f"**Resposta:**\n{formatted_response}", unsafe_allow_html=True)

def main():
    st.set_page_config("Demite Cartório")
    st.header("IA criada para o Arthur manso testar")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_pdfs_and_create_index()
 #       if st.session_state.vector_store:
 #           st.success("PDFs carregados e índice criado com sucesso!")

    user_question = st.text_input("Faça uma pergunta com base nos docs enviados para o tesquinho: ")

    if user_question and st.session_state.vector_store:
        user_input(user_question, st.session_state.vector_store)

if __name__ == "__main__":
    main()
