import os
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from helpers import get_repo_config, load_documents_from_repo

# Load environment variables
load_dotenv()

def main():
    # Configuration
    github_token = os.getenv("GITHUB_TOKEN")
    voyage_api_key = os.getenv("VOYAGE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    all_documents = []

    # Load issues from up to 3 repositories
    for i in range(1, 4):
        owner, repo, branch = get_repo_config(i)
        if owner and repo:
            print(f"Loading issues from repository {i}: {owner}/{repo}")
            documents = load_documents_from_repo(github_token, owner, repo, branch)
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} documents from repository {i}")
        else:
            print(f"Repository {i} not configured, skipping.")

    if not all_documents:
        print("No documents loaded. Please check your .env configuration.")
        return

    print(f"Total documents loaded: {len(all_documents)}")

    # Print content of loaded documents
    for i, doc in enumerate(all_documents):
        print(f"Document {i + 1} content: {doc.page_content[:100]}...") # Print first 100 characters

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = text_splitter.split_documents(all_documents)
    print(f"Total splits after text splitting: {len(splits)}")

    if not splits:
        print("No text splits generated. Using original documents.")
        splits = all_documents

    # Create embeddings using VoyageAI
    embeddings = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model="voyage-code-2")

    # Create FAISS index
    try:
        texts = [doc.page_content for doc in splits]
        vectorstore = FAISS.from_texts(texts, embeddings)
        print("FAISS index created successfully")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    # Example query
    query = "What are the most common issues in this repository?"
    try:
        response = qa_chain.invoke(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error during query: {e}")

    # Demonstrate retrieval of top document
    try:
        result = vectorstore.similarity_search(query, k=1)
        if result:
            top1_retrieved_doc = result[0].page_content
            print(f"\nTop retrieved document: {top1_retrieved_doc}")
        else:
            print("\nNo relevant documents found.")
    except Exception as e:
        print(f"Error during document retrieval: {e}")

if __name__ == "__main__":
    main()