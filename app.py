import os
import streamlit as st
from dotenv import load_dotenv

from app.utils.helpers import check_environment
from app.utils.errors import (
    InvalidURLError,
    TranscriptNotAvailableError,
    VideoUnavailableError,
    VectorStoreError,
    LLMError,
    RAGException,
)

load_dotenv()
check_environment()


# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "retriever_config" not in st.session_state:
    st.session_state.retriever_config = None

if "memory" not in st.session_state:
    st.session_state.memory = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "reranker" not in st.session_state:
    st.session_state.reranker = None

if "video_metadata" not in st.session_state:
    st.session_state.video_metadata = {}

if "chat_history_display" not in st.session_state:
    st.session_state.chat_history_display = []

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "processing" not in st.session_state:
    st.session_state.processing = False

if "use_reranking" not in st.session_state:
    st.session_state.use_reranking = False

if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = 5


# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="YouTube RAG Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown(
    """
<style>
    /* Make chat messages look cleaner */
    .stChatMessage {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
    }
    
    /* Source expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Make the sidebar header cleaner */
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Hide the Streamlit default menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# SIDEBAR IMPLEMENTATION
# =====================================================================
with st.sidebar:
    # SECTION 1 — App header
    st.title("🎬 YouTube RAG")
    st.caption("Chat with any YouTube video using AI")
    st.divider()

    # SECTION 2 — URL input and processing
    st.subheader("📹 Load a Video")
    youtube_url = st.text_input(
        label="YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        key="youtube_url_input",
        help="Paste any YouTube video URL with captions",
    )
    process_button = st.button(
        "⚡ Process Video",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.processing,
    )

    # VIDEO PROCESSING LOGIC
    if process_button and youtube_url:
        clean_url = youtube_url.strip()
        # Step 1 — Validate URL format
        if "youtube.com" not in clean_url and "youtu.be" not in clean_url:
            st.sidebar.error("Please enter a valid YouTube URL.")
        else:
            # Step 2 — Set processing state
            st.session_state.processing = True
            st.session_state.video_processed = False
            st.session_state.chat_history_display = []

            # Step 3 — Run indexing pipeline with a progress display
            with st.status("Processing video...", expanded=True) as status:
                try:
                    st.write("📥 Fetching transcript...")
                    from app.retrieval.vectorstore import index_video

                    vectorstore, video_metadata = index_video(clean_url)
                    st.write("✅ Transcript fetched and indexed.")

                    st.write("🔗 Building retriever...")
                    from app.retrieval.retriever import (
                        build_retriever,
                        build_retriever_with_reranking,
                    )
                    from app.utils.helpers import (
                        get_groq_llm,
                        get_reranker_model,
                    )

                    llm = get_groq_llm()
                    reranker = get_reranker_model()
                    retriever = build_retriever(
                        vectorstore, k=st.session_state.retrieval_k
                    )
                    retriever_config = build_retriever_with_reranking(
                        vectorstore,
                        reranker,
                        initial_k=st.session_state.retrieval_k * 4,
                        final_k=st.session_state.retrieval_k,
                    )
                    st.write("✅ Retriever ready.")

                    st.write("🧠 Initializing conversation memory...")
                    from app.generation.generator import create_conversation_memory

                    memory = create_conversation_memory()
                    st.write("✅ Memory initialized.")

                    # Store everything in session state
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = retriever
                    st.session_state.retriever_config = retriever_config
                    st.session_state.memory = memory
                    st.session_state.llm = llm
                    st.session_state.reranker = reranker
                    st.session_state.video_metadata = video_metadata
                    st.session_state.video_processed = True
                    st.session_state.processing = False

                    status.update(
                        label="✅ Video ready!",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()

                except InvalidURLError as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error("❌ Invalid URL: Please check your YouTube link.")
                except TranscriptNotAvailableError as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error("❌ No captions: This video has no transcript available.")
                except VideoUnavailableError as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error("❌ Video unavailable: Check if the video is public.")
                except VectorStoreError as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error("❌ Index error: Could not build the search index.")
                except LLMError as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error("❌ AI error: Check your GROQ_API_KEY in the .env file.")
                except RAGException as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error(f"❌ Error: {str(e)}")
                except Exception as e:
                    st.session_state.processing = False
                    status.update(label="❌ Processing failed", state="error")
                    st.sidebar.error(f"❌ Unexpected error: {str(e)}")

    elif process_button and not youtube_url:
        st.sidebar.warning("Please enter a YouTube URL first.")

    # SECTION 3 — Video info (only shown after processing)
    st.divider()
    st.subheader("📊 Video Info")
    if st.session_state.video_processed:
        metadata = st.session_state.video_metadata
        title = metadata.get("title", "Unknown Title")
        channel = metadata.get("channel", "Unknown Channel")
        duration_seconds = metadata.get("duration_seconds", 0)
        total_chunks = metadata.get("total_chunks", 0)

        duration_formatted = (
            f"{int(duration_seconds // 60)} min"
            if duration_seconds
            else "N/A"
        )

        st.markdown(f"**Title:** {title}")
        st.markdown(f"**Channel:** {channel}")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Duration", duration_formatted)
        with m_col2:
            st.metric("Chunks", total_chunks)

        st.success("✅ Ready to chat")
    else:
        st.info("No video loaded yet.")

    # SECTION 4 — Settings
    st.divider()
    st.subheader("⚙️ Settings")
    use_reranking = st.toggle(
        "Enable reranking",
        value=st.session_state.use_reranking,
        help="Reranking improves answer quality but is slightly slower. Uses cross-encoder/ms-marco-MiniLM-L-6-v2.",
    )
    st.session_state.use_reranking = use_reranking

    retrieval_k = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=10,
        value=st.session_state.retrieval_k,
        step=1,
        help="How many transcript chunks to retrieve per question. Higher = more context but slower.",
    )
    if retrieval_k != st.session_state.retrieval_k:
        st.session_state.retrieval_k = retrieval_k
        if st.session_state.vectorstore is not None:
            from app.retrieval.retriever import (
                build_retriever,
                build_retriever_with_reranking,
            )

            st.session_state.retriever = build_retriever(
                st.session_state.vectorstore, k=retrieval_k
            )
            if st.session_state.reranker is not None:
                st.session_state.retriever_config = build_retriever_with_reranking(
                    st.session_state.vectorstore,
                    st.session_state.reranker,
                    initial_k=retrieval_k * 4,
                    final_k=retrieval_k,
                )

    if st.session_state.video_processed:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history_display = []
            if st.session_state.memory:
                from app.generation.generator import clear_memory

                clear_memory(st.session_state.memory)
            st.rerun()

    # SECTION 5 — Footer
    st.divider()
    st.caption("Powered by Groq · LLaMA 3.3 70B")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector Store: FAISS")


# =====================================================================
# MAIN CHAT AREA
# =====================================================================

# SECTION 1 — App header
st.title("💬 YouTube RAG Assistant")
if st.session_state.video_processed:
    video_title = st.session_state.video_metadata.get("title", "Video")
    st.caption(f"Chatting with: **{video_title}**")
else:
    st.caption("Load a YouTube video from the sidebar to begin.")

tab1, tab2 = st.tabs(["💬 Chat", "📊 Evaluation"])

with tab1:
    # SECTION 2 — Chat history display
    chat_container = st.container()

    with chat_container:
        if (
            not st.session_state.chat_history_display
            and not st.session_state.video_processed
        ):
            st.info(
                "👈 **Get started:**\n\n"
                "1. Paste a YouTube URL in the sidebar\n"
                "2. Click **Process Video**\n"
                "3. Ask any question about the video"
            )
        elif (
            not st.session_state.chat_history_display
            and st.session_state.video_processed
        ):
            video_title = st.session_state.video_metadata.get("title", "Video")
            st.info(
                f"✅ **{video_title}** is ready!\n\n"
                "Ask me anything about this video."
            )
        else:
            for message in st.session_state.chat_history_display:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "user":
                    with st.chat_message("user"):
                        st.write(content)
                elif role == "assistant":
                    with st.chat_message("assistant"):
                        if message.get("not_found"):
                            st.warning(content)
                        else:
                            st.write(content)

                        sources_md = message.get("sources_markdown")
                        if sources_md:
                            with st.expander("📍 View sources", expanded=False):
                                st.markdown(sources_md)

    # SECTION 3 — Chat input area & ANSWER GENERATION LOGIC
    with st.container():
        st.divider()

        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.chat_input(
                placeholder="Ask a question about the video...",
                disabled=not st.session_state.video_processed,
                key="chat_input",
            )

        if user_input and st.session_state.video_processed:
            # Step 1 — Add user message to display history
            st.session_state.chat_history_display.append(
                {
                    "role": "user",
                    "content": user_input,
                    "sources_markdown": None,
                    "not_found": False,
                }
            )

            # Step 2 — Generate answer with spinner
            with st.spinner("🤔 Thinking..."):
                try:
                    from app.generation.generator import (
                        generate_answer_with_sources,
                    )

                    if st.session_state.use_reranking:
                        result = generate_answer_with_sources(
                            query=user_input,
                            retriever_or_config=st.session_state.retriever_config,
                            memory=st.session_state.memory,
                            llm=st.session_state.llm,
                            use_reranking=True,
                        )
                    else:
                        result = generate_answer_with_sources(
                            query=user_input,
                            retriever_or_config=st.session_state.retriever,
                            memory=st.session_state.memory,
                            llm=st.session_state.llm,
                            use_reranking=False,
                        )

                    # Step 3 — Add AI response to display history
                    st.session_state.chat_history_display.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources_markdown": result["sources_markdown"],
                            "not_found": result["not_found"],
                        }
                    )

                except Exception as e:
                    err_text = str(e)
                    err_lower = err_text.lower()
                    if "401" in err_lower or "api key" in err_lower:
                        display_err = (
                            "Groq API error. Check your API key in .env."
                        )
                    else:
                        display_err = f"Sorry, something went wrong: {err_text}"

                    st.session_state.chat_history_display.append(
                        {
                            "role": "assistant",
                            "content": display_err,
                            "sources_markdown": None,
                            "not_found": False,
                        }
                    )

            # Step 4 — Rerun to refresh the chat display
            st.rerun()

with tab2:
    st.header("📊 RAG Evaluation")
    st.caption("Compare different retrieval configurations")

    if not st.session_state.video_processed:
        st.info("Process a video first to run evaluation.")
    else:
        st.subheader("Quick Retrieval Test")
        st.write(
            "Test a specific question and see what chunks "
            "are retrieved before the LLM generates an answer."
        )

        test_query = st.text_input(
            "Test query",
            placeholder="Enter a question to test retrieval...",
        )

        col1, col2 = st.columns(2)
        with col1:
            test_k = st.slider("k (chunks)", 2, 10, 5, key="eval_k")
        with col2:
            test_rerank = st.toggle("Use reranking", key="eval_rerank")

        if st.button("🔍 Test Retrieval", type="primary"):
            if test_query:
                with st.spinner("Retrieving..."):
                    try:
                        from app.retrieval.retriever import (
                            build_retriever,
                            build_retriever_with_reranking,
                            retrieve_and_format,
                            retrieve_and_format_with_reranking,
                        )

                        retriever = build_retriever(
                            st.session_state.vectorstore, k=test_k
                        )

                        if test_rerank:
                            if st.session_state.reranker is None:
                                from app.utils.helpers import get_reranker_model

                                st.session_state.reranker = get_reranker_model()

                            config = build_retriever_with_reranking(
                                st.session_state.vectorstore,
                                st.session_state.reranker,
                                initial_k=test_k * 4,
                                final_k=test_k,
                            )
                            docs, sources = retrieve_and_format_with_reranking(
                                test_query, config
                            )
                        else:
                            docs, sources = retrieve_and_format(
                                retriever, test_query
                            )

                        st.subheader(f"Retrieved {len(sources)} chunks:")

                        for i, source in enumerate(sources):
                            with st.expander(
                                f"Chunk {source['chunk_id']} — "
                                f"{source['start_timestamp_formatted']}",
                                expanded=(i == 0),
                            ):
                                st.write(source["text"])
                                if source.get("timestamp_url"):
                                    st.markdown(
                                        f"[▶ Jump to {source['start_timestamp_formatted']} in video]"
                                        f"({source['timestamp_url']})"
                                    )
                                if source.get("reranker_score"):
                                    st.caption(
                                        f"Reranker score: "
                                        f"{source.get('reranker_score', 'N/A')}"
                                    )
                    except Exception as e:
                        st.error(f"Retrieval test failed: {str(e)}")
            else:
                st.warning("Enter a test query first.")

        st.divider()
        st.subheader("Full Evaluation")
        st.write("Run the full evaluation suite from the terminal:")
        st.code("python evaluation/run_eval.py", language="bash")
        st.caption(
            "This tests 3 configurations × 8 questions "
            "and saves results to evaluation/eval_results.csv"
        )

