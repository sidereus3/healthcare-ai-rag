import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def load_and_preprocess_data(csv_path: str):
    """
    Reads the healthcare data and constructs unified Document objects
    enriched with operational metadata.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean step: Drop rows where the critical clinical narrative is missing
    df = df.dropna(subset=["transcription"])

    documents = []
    for idx, row in df.iterrows():
        # Combine clinical specialties and descriptions to give the vector index more context
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
    """
    Splits larger medical files into small, precise tokens to prevent
    symptom dilution during vector similarity matching.
    """
    print("Chunking documents into segments...")
    # 1000 characters with a 200 character overlap ensures sentences
    # aren't abruptly sliced in half across chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks ready for embeddings.")
    return chunks


if __name__ == "__main__":
    # temporary location
    csv_file = "mtsamples.csv"

    if not os.path.exists(csv_file):
        print(f"Error: Could not find '{csv_file}' at the root level.")
    else:
        raw_docs = load_and_preprocess_data(csv_file)
        chunked_docs = chunk_documents(raw_docs)

        # Output verification
        print("\n--- Verifying Sample Chunk [0] ---")
        if chunked_docs:
            print(chunked_docs[0].page_content)
