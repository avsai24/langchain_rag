import yaml
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="---------- %(levelname)s - %(message)s ----------", 
)

def web_scrape():
    with open("/Users/venkatasaiancha/Desktop/lanchain_rag/links.yaml", "r") as file:
        data = yaml.safe_load(file)
    urls = [entry for entry in data["links"]]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs

def cleaning(docs):
    html_cleaner = Html2TextTransformer()
    clean_docs = html_cleaner.transform_documents(docs)
    return clean_docs

def split_text(clean_docs):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(clean_docs)
    return all_splits

def store_embeddings(splitted_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vectorstore = Chroma.from_documents(splitted_docs, embeddings, persist_directory="/Users/venkatasaiancha/Desktop/lanchain_rag/chroma_db")

def main():
    logging.info("entered webscraping")
    doccuments = web_scrape()
    logging.info("entered cleaning")
    cleaned_docs = cleaning(doccuments)
    logging.info("entered text splitting")
    splitted_docs = split_text(cleaned_docs)
    logging.info("entered embeddings and storing in vectordb")
    store_embeddings(splitted_docs)
    logging.info("Embeddings stored successfully in ChromaDB")

if __name__ == "__main__":
    main()