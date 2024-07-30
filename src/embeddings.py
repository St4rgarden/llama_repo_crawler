from langchain_voyageai import VoyageAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_embeddings_and_vectorstore(documents, voyage_api_key):
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(documents)
    print(f"Total splits after text splitting: {len(splits)}")

    if not splits:
        print("No text splits generated. Using original documents.")
        splits = documents

    # Create embeddings using VoyageAI
    embeddings = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model="voyage-code-2")

    # Create FAISS index
    try:
        texts = [doc.page_content for doc in splits]
        vectorstore = FAISS.from_texts(texts, embeddings)
        print("FAISS index created successfully")
        return vectorstore
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None