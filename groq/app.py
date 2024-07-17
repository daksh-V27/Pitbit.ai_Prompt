import streamlit as st
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Set your Groq API key as a string
groq_api_key = "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"

# Initialize Streamlit components and session state
st.title("Text_Maven")
st.write("---------------------------------------------------------------------------")   

# Initialize ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Your name is Maven the textbot :)
    your developer name is Daksh Verma
    Answer the questions based on the provided context and take into account the previously saved context only when very necessary.
    If you can answer from the current text only use it.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

def create_vector_embedding(text):
    if "all_text" not in st.session_state:
        st.session_state.all_text = ""
    
    st.session_state.all_text += " " + text
    
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    st.session_state.final_documents = st.session_state.text_splitter.split_text(st.session_state.all_text)
    

    if len(st.session_state.final_documents) == 0:
        st.error("No document chunks created. Please check the text splitter settings.")
        return

    st.session_state.vectors = FAISS.from_texts(st.session_state.final_documents, st.session_state.embeddings)
    st.write("I am ready to answer your Questions. I do remember previous saved texts : )")

user_text = st.text_area("Enter the entire document", height=200)

if st.button("SAVE"):
    if user_text.strip() == "":
        st.error("Cant keep it empty!!.")
    else:
        create_vector_embedding(user_text)

st.write("---------------------------------------------------------------------------")   

user_prompt = st.text_input("Enter your question")

if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please save your story first !!.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        response_time = time.process_time() - start
        print(f"Response time: {response_time}")

        st.write(response['answer'])
        
st.write("---------------------------------------------------------------------------")
