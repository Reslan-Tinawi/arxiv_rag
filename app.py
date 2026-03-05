from dotenv import load_dotenv
import streamlit as st
import boto3
import os

# Updated LangChain Imports to avoid deprecation warnings
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from botocore.config import Config

# --- Setup & Configuration ---
load_dotenv()

# We use os.getenv with fallbacks to avoid silent failures
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# st.set_page_config must be the first Streamlit command
st.set_page_config(page_title="ArXiv AI Synthesizer", page_icon="📚", layout="wide")


@st.cache_resource
def initialize_services():
    """Initializes AWS and Pinecone clients, cached to prevent re-runs."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing from environment variables.")

    retry_config = Config(
        region_name=AWS_REGION, retries={"max_attempts": 5, "mode": "standard"}
    )

    boto_session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

    bedrock_client = boto_session.client(
        service_name="bedrock-runtime", config=retry_config
    )

    # Now using langchain_aws instead of langchain_community
    embeddings = BedrockEmbeddings(
        client=bedrock_client, model_id="amazon.titan-embed-text-v1"
    )

    llm = ChatBedrock(
        client=bedrock_client,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",  # we can experiment with different models here
        model_kwargs={"temperature": 0.2},
    )

    vectorstore = PineconeVectorStore(
        index_name="arxiv-cs-methodologies",
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
        text_key="text",  # Explicitly map to the 'text' field shown in your Pinecone record
    )

    return llm, vectorstore


def main():
    st.title("📚 Temporal RAG: ArXiv Methodology Synthesizer")

    try:
        llm, vectorstore = initialize_services()
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        st.stop()  # Stop execution if services fail

    # --- Sidebar: Temporal and Categorical Filters ---
    st.sidebar.header("Search Filters")

    min_year, max_year = 1990, 2026
    selected_years = st.sidebar.slider(
        "Select Publication Date Range",
        min_value=min_year,
        max_value=max_year,
        value=(2020, max_year),
    )

    start_date_int = selected_years[0] * 10000 + 101
    end_date_int = selected_years[1] * 10000 + 1231

    categories = ["cs.AI", "cs.CL", "cs.CV", "cs.DB"]
    selected_cats = st.sidebar.multiselect(
        "Filter by Categories", categories, default=categories
    )

    # Prevent Pinecone error if all categories are deselected
    if not selected_cats:
        st.sidebar.warning("Please select at least one category to search.")
        st.stop()

    search_filter = {
        "update_date": {"$gte": start_date_int, "$lte": end_date_int},
        "categories": {"$in": selected_cats},
    }

    # --- Modern Chat Interface ---
    # Store chat history in session state for a better Jupyter/App feel
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input(
        "Ask about CS methodologies (e.g., 'Recent approaches to document segmentation?')"
    ):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching literature and generating synthesis..."):
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": search_filter}
                )

                system_prompt = (
                    "You are an expert Computer Science researcher. Use the following pieces of "
                    "retrieved abstract context to answer the user's question. "
                    "Always cite the 'title' and 'date' of the papers you use in your answer.\n\n"
                    "Context: {context}"
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ]
                )

                # Modern LCEL approach: Format documents manually to include metadata
                def format_docs(docs):
                    return "\n\n".join(
                        f"[Title: {doc.metadata.get('title', 'Unknown')}]\n"
                        f"[Date: {doc.metadata.get('date_display', 'Unknown')}]\n"
                        f"Abstract: {doc.page_content}"
                        for doc in docs
                    )

                # 1. Setup Retrieval & Input Passthrough
                retrieval_setup = RunnableParallel(
                    {"context": retriever, "input": RunnablePassthrough()}
                )

                # 2. Setup the LLM Chain (Formatting -> Prompt -> LLM -> String)
                answer_chain = (
                    {
                        "context": lambda x: format_docs(x["context"]),
                        "input": lambda x: x["input"],
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # 3. Combine to return BOTH the generated answer and the raw source documents
                rag_chain = retrieval_setup | RunnableParallel(
                    {"answer": answer_chain, "context": lambda x: x["context"]}
                )

                try:
                    # Note: LCEL allows us to pass the string query directly
                    response = rag_chain.invoke(query)
                    answer = response["answer"]
                    docs = response["context"]

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                    # Render sources
                    if docs:
                        st.markdown("### 📚 Sources Retrieved")
                        for i, doc in enumerate(docs):
                            title = doc.metadata.get("title", "Unknown Title")
                            date = doc.metadata.get("date_display", "Unknown Date")
                            cats = doc.metadata.get("categories", ["Unknown"])

                            with st.expander(f"{i + 1}. {title} ({date})"):
                                st.caption(f"**Categories:** {', '.join(cats)}")
                                st.write(doc.page_content)
                except Exception as e:
                    st.error(f"An error occurred during retrieval: {e}")


if __name__ == "__main__":
    main()
