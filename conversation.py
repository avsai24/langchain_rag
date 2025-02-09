from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import logging
import time

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(collection_name="conversation_history", embedding_function=embeddings, persist_directory="/Users/venkatasaiancha/Desktop/lanchain_rag/chroma_db")

def add_conversation(user_input, model_response, model_name):

    timestamp = int(time.time()) 
    doc = Document(
        page_content=f"User: {user_input}\nModel: {model_response}",
        metadata={"user_input": user_input, "model_response": model_response,"model_name": model_name, "timestamp": timestamp}
    )
    chroma_db.add_documents([doc])


def load_conversations():
   
    documents = chroma_db._collection.get(include=["metadatas"]) 
    if not documents["metadatas"]:
        logging.info("No conversations found in the database.")
        return []
    conversations = [
        {
            "user": meta["user_input"],
            "model": meta["model_response"],
            "model_name": meta["model_name"],
            "timestamp": meta.get("timestamp", 0) 
        }
        for meta in documents["metadatas"]
    ]
    return sorted(conversations, key=lambda x: x["timestamp"])

def print_conversations():
    conversations = load_conversations()
    for conversation in conversations:
        timestamp = conversation.get("timestamp")  
        if timestamp:
            readable_time = time.ctime(timestamp)  
        else:
            readable_time = "No timestamp available"
        print(f"User: {conversation['user']}, Model: {conversation['model']}, Timestamp: {readable_time}")
        print("-" * 50)  


def delete_conversations():
    try:
        document_ids = chroma_db._collection.get()["ids"]        
        if not document_ids:
            logging.warning("No conversations to delete.")
            return
        
        chroma_db.delete(ids=document_ids)  
        logging.info("All conversations deleted successfully.")
    except Exception as e:
        logging.error(f"Error deleting conversations: {e}")

if __name__ == "__main__":
    logging.info("adding question1.")
    add_conversation("question 1", "answer 1","llama3.2")
    logging.info("adding question2.")
    add_conversation("question 2", "answer 2","deepseek-r1")
    logging.info("loading conversations")
    load_conversations()
    logging.info("adding question3.")
    add_conversation("question 3", "answer 3","llama3.2")
    logging.info("loading conversations.")
    load_conversations()
    logging.info("printing conversations.")
    print_conversations()   
    logging.info("deleting conversations.")
    delete_conversations()
    logging.info("printing conversations.")
    print_conversations()
    delete_conversations()