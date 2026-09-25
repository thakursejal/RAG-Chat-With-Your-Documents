import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import InferenceClient

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Chat With Your Documents",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat With Your Documents")
st.caption("RAG-powered document question answering")

# =========================================================
# HUGGING FACE CLIENT
# =========================================================

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    api_key=HF_TOKEN
)

MODEL = "openai/gpt-oss-120b:fastest"

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(text, chunk_size=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if chunk.strip():
            chunks.append(chunk)

    return chunks

# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text

# =========================================================
# RAG PIPELINE
# =========================================================

def create_vector_database(chunks):

    embeddings = embedding_model.encode(chunks)

    chroma_client = chromadb.Client()

    collection = chroma_client.create_collection(
        name="document_knowledge"
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return chroma_client, collection


def ask_rag(question, collection):

    question_embedding = embedding_model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=min(3, collection.count())
    )

    retrieved_documents = results["documents"][0]

    context = "\n\n".join(retrieved_documents)

    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not available in the context, say:

"The information is not available in the uploaded document."

Do not invent information.

Document Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300
    )

    answer = response.choices[0].message.content

    return answer, retrieved_documents

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    st.markdown("---")

    st.info(
        "Your document is processed only after you "
        "explicitly upload it."
    )

# =========================================================
# DOCUMENT PROCESSING
# =========================================================

if uploaded_file is not None:

    if "processed_file" not in st.session_state:

        with st.spinner("📖 Reading your document..."):

            document_text = extract_pdf_text(uploaded_file)

        if not document_text.strip():

            st.error(
                "❌ No readable text was found in this PDF."
            )

            st.stop()

        with st.spinner("✂️ Creating document chunks..."):

            chunks = create_chunks(document_text)

        with st.spinner("🧠 Creating embeddings and vector database..."):

            chroma_client, collection = create_vector_database(chunks)

        st.session_state.collection = collection
        st.session_state.processed_file = uploaded_file.name

        st.success(
            f"✅ Document processed successfully! "
            f"Created {len(chunks)} chunks."
        )

    else:

        collection = st.session_state.collection

        st.success(
            f"✅ Ready to answer questions about "
            f"**{st.session_state.processed_file}**"
        )

    # =====================================================
    # QUESTION SECTION
    # =====================================================

    st.subheader("💬 Ask Questions About Your Document")

    question = st.text_input(
        "Enter your question:",
        placeholder="What is this document about?"
    )

    if st.button("🔎 Ask AI", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("🤖 Searching document and generating answer..."):

                answer, retrieved_documents = ask_rag(
                    question,
                    collection
                )

            st.subheader("💡 Answer")

            st.write(answer)

            with st.expander("📚 View Retrieved Context"):

                for i, document in enumerate(
                    retrieved_documents
                ):

                    st.markdown(
                        f"**Retrieved Chunk {i + 1}**"
                    )

                    st.write(document)

                    st.markdown("---")

else:

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting "
        "with your document."
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Built by Thakur Sejal | Codomax Digital Solutions Internship | Module 4"
)
