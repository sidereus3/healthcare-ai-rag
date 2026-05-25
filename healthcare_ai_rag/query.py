import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

load_dotenv()


def initialize_rag_chain(db_path="faiss_index"):
    """
    Loads the local FAISS index and constructs a LangChain Expression Language (LCEL)
    chain linking retrieval directly to the Azure instance (GPT-4o for now).
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Could not find local vector database at '{db_path}/'. "
            "Please run 'pipeline.py' first to initialize the index."
        )

    print("Loading vector database and initializing Azure services...")

    # re-initialize the same embedding client used to create the index
    embeddings_client = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # load the index from disk
    vector_db = FAISS.load_local(
        db_path, embeddings_client, allow_dangerous_deserialization=True
    )

    # configure the index to act as a retriever (pulling the top 3 matches)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # initialize the Azure OpenAI Chat client for GPT-4o
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.1,  # Low temperature keeps the model objective and grounded in the text
    )

    # define a strict clinical assistant system prompt
    # This prevents hallucinations by forcing compliance with the provided context
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an advanced clinical analytics assistant. Your job is to answer queries "
                    "using only the provided medical context fragments. \n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "1. Base your answer STRICTLY on the context provided below. Do not extrapolate.\n"
                    "2. If the context does not contain enough information to conclusively answer, "
                    "state clearly: 'I am sorry, but the provided medical history does not contain "
                    "sufficient evidence to answer this question.'\n\n"
                    "--- PROVIDED MEDICAL CONTEXT ---\n"
                    "{context}\n"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        """Helper to stitch retrieved text blocks into a clean multi-document string."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # sssemble the RAG pipeline using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    # Test clinical lookup query
    query = "What symptoms are associated with acute allergic reactions in the dataset?"

    try:
        chain = initialize_rag_chain()
        print(f"\nSending Query to RAG System: '{query}'\n")
        print("--- GPT-4o Grounded Response ---")

        # Invoke the complete end-to-end chain
        response = chain.invoke(query)
        print(response)

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
