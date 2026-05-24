import os

import pandas as pd
from dotenv import load_dotenv
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
    """
    Initializes the Azure OpenAI Embedding connection using configuration
    pulled directly from the local environment file.
    """
    print("Initializing Azure OpenAI Embeddings client...")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    return embeddings


if __name__ == "__main__":
    csv_file = "mtsamples.csv"

    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}' at the root level.")
    else:
        raw_docs = load_and_preprocess_data(csv_file)
        chunked_docs = chunk_documents(raw_docs)
        embeddings_client = initialize_embeddings()

        # Test a single embedding vector to confirm connectivity with Azure
        print("Testing Azure connectivity by embedding a sample chunk...")
        sample_vector = embeddings_client.embed_query(chunked_docs[0].page_content)
        print(f"Success! Generated vector embedding of length: {len(sample_vector)}")
