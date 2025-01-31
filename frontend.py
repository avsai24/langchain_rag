import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from conversation import add_conversation, delete_conversations, load_conversations
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", )

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="/Users/venkatasaiancha/Desktop/lanchain_rag/chroma_db", embedding_function=embeddings)

def model_name():
    options = ["llama3.2", "deepseek-r1"]
    st.sidebar.selectbox("Choose a model", options, key="model_name")

def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = load_conversations()

def render_sidebar():
    model_name()
    if st.sidebar.button("Delete All Conversations"):
        delete_conversations()
        st.session_state.conversation = []  
        st.sidebar.success("All conversations deleted successfully.")
    
    if st.sidebar.button("View Session Data"):
        st.sidebar.write("### Current Session Data:")
        st.sidebar.write(st.session_state)

def display_conversation_history():
    logging.info("entered display_conversation history")
    for entry in st.session_state.conversation:
        with st.chat_message("user"):
            st.markdown(f"{entry['user']}") 

        with st.chat_message("assistant"):
            st.markdown(f"{entry['model']}")

def render_ui():
    st.title("RAG Using Streamlit & Ollama")
    st.write("---")
    render_sidebar()
    logging.info(f"rendered_sidebar")
    display_conversation_history()
    logging.info(f"rendered display_conversation")
    chat_input_handler() 
    logging.info("entered chat_input_handler")

def chat_input_handler():
    prompt = st.chat_input("Type your message...")
    if prompt:  
        handle_generate_response(prompt)

def handle_generate_response(prompt):
    logging.info("entered handle_generate_response")
    if not prompt.strip():
        st.warning("Please enter a prompt to generate a response.")
        return
    model_name = st.session_state.model_name
    retrieved_docs = vectorstore.similarity_search(prompt, k=3)
    chat_history = [f"User: {entry['user']}\nModel: {entry['model']}" for entry in st.session_state.conversation]
    
    prompt_template = PromptTemplate(
    input_variables=["retrieved_docs", "chat_history", "prompt"],
    template="""
    ### **Context to Use:**
    The following retrieved documents contain factual information:
    {retrieved_docs}

    ### **Previous Chat History:**
    Chat history helps maintain conversation flow but should NOT be used for factual information:
    {chat_history}

    ### **Instructions:**
    **Use ONLY the retrieved documents** to generate your answer.
    **DO NOT make up any information.** Your response must be STRICTLY based on the retrieved documents.
    **Chat history is only for reference, NOT for providing facts.**
    **If the retrieved documents do not provide sufficient information to answer the question, respond with:**  
        `"Sorry, I cannot provide an answer based on the given information."`
    **Do NOT infer, assume, or generate unrelated details.**
    
    ### **User's Question:**
    {prompt}

    ### **Final Answer (Strictly Based on Context Above):**
    """
    )

    llm = OllamaLLM(model=model_name)
    prompt_temp = prompt_template.format(retrieved_docs=retrieved_docs,chat_history=chat_history,prompt=prompt)
    with st.chat_message("user"):
        st.markdown(f"{prompt}") 
    response = llm.invoke(prompt_temp)
    add_conversation(prompt,response,model_name)
    st.session_state.conversation.append({"user": prompt, "model": response, "model_name":model_name})
    with st.chat_message("assistant"):
        st.markdown(f"{response}")

def main():
    logging.info("session start")
    initialize_session_state()
    logging.info("initialized session state")
    render_ui()
    logging.info("rendered ui")
    
if __name__ == "__main__":
    main()