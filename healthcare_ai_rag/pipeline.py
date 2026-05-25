import os

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_and_preprocess_data(csv_path: str):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["transcription"])

    documents = []
    for idx, row in df.iterrows():
        context_text = (
            f"Medical Specialty: {row['medical_specialty']}\n"
            f"Description: {row['description']}\n"
            f"Transcription:\n{row['transcription']}"
        )

        doc = Document(
            page_content=context_text,
            metadata={
                "source_row": idx,
                "specialty": str(row["medical_specialty"]),
                "sample_name": str(row["sample_name"]),
            },
        )
        documents.append(doc)

    print(f"Processed {len(documents)} source medical records.")
    return documents


def chunk_documents(documents: list):
    print("Chunking documents into segments...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks ready for embeddings.")
    return chunks


def initialize_embeddings():
    print("Initializing Azure OpenAI Embeddings client...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    return embeddings


def build_or_load_vector_store(chunks, embeddings_client, db_path="faiss_index"):
    """
    Builds a local FAISS index from document chunks if it doesn't exist.
    Otherwise, loads the index directly from disk to optimize performance.
    """
    # Allow dangerous deserialization since this is a local, trusted database index file
    if os.path.exists(db_path):
        print(f"Found existing local vector database at '{db_path}/'. Loading index...")
        db = FAISS.load_local(
            db_path, embeddings_client, allow_dangerous_deserialization=True
        )
    else:
        print(
            "No local database found. Generating embeddings and building FAISS index..."
        )

        print(f"Embedding all {len(chunks)} chunks into vector store...")
        db = FAISS.from_documents(chunks, embeddings_client)

        # Save to local workspace directory
        db.save_local(db_path)
        print(f"Vector database built and saved locally to '{db_path}/'.")

    return db


if __name__ == "__main__":
    csv_file = "mtsamples.csv"

    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}' at the root level.")
    else:
        # Preprocessing & setup
        raw_docs = load_and_preprocess_data(csv_file)
        chunked_docs = chunk_documents(raw_docs)
        embeddings_client = initialize_embeddings()

        # vector DB compilation
        vector_db = build_or_load_vector_store(chunked_docs, embeddings_client)

        # Run a quick test query to confirm similarity search works
        test_query = "What symptoms are associated with acute allergic reactions?"
        print(f"\nExecuting semantic similarity search test for query: '{test_query}'")

        # Retrieve the top 2 most mathematically similar text chunks
        results = vector_db.similarity_search(test_query, k=2)

        print("\n--- Top Similarity Search Result Match ---")
        if results:
            print(f"Source Speciality: {results[0].metadata['specialty']}")
            print(f"Snippet:\n{results[0].page_content[:400]}...")
