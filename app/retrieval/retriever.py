"""
Retrieval module.
Wraps the FAISS vector store in a LangChain retriever.
Handles top-k semantic retrieval, MMR diversification, deduplication,
cross-encoder reranking, and formats retrieved chunks with timestamps
and YouTube deep-link URLs.
"""

from dotenv import load_dotenv

from app.utils.errors import RetrievalError
from app.utils.logger import get_logger

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

logger = get_logger(__name__)
load_dotenv()


def build_retriever(
    vectorstore,
    search_type: str = "similarity",
    k: int = 5,
):
    """
    Wrap the FAISS vector store in a LangChain VectorStoreRetriever.
    """
    if search_type == "similarity":
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
    elif search_type == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 4,
                "lambda_mult": 0.7,
            },
        )
    else:
        raise ValueError(
            f"Invalid search_type '{search_type}'. Use 'similarity' or 'mmr'."
        )

    logger.info("Retriever built — type: %s, k: %d", search_type, k)
    return retriever


def retrieve_chunks(retriever, query: str) -> list:
    """
    Run a retrieval query and return the list of relevant Document chunks.
    """
    display_query = query[:80] + "..." if len(query) > 80 else query
    logger.info("Retrieving chunks for query: '%s'", display_query)

    try:
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        else:
            docs = retriever.get_relevant_documents(query)

        logger.info("Retrieved %d chunks.", len(docs))
        return docs
    except Exception as e:
        logger.error("Retrieval failed for query '%s': %s", query, str(e))
        raise RetrievalError(query, str(e))


def format_retrieved_sources(documents: list) -> list[dict]:
    """
    Convert a list of retrieved LangChain Documents into structured dictionaries.
    """
    if not documents:
        return []

    sources = []
    for doc in documents:
        meta = getattr(doc, "metadata", {})
        content = getattr(doc, "page_content", "")
        formatted_ts = meta.get("start_timestamp_formatted", "0:00")
        title = meta.get("title", "Unknown")

        source_dict = {
            "chunk_id": meta.get("chunk_id", "unknown"),
            "text": content,
            "text_preview": (
                content[:200] + "..." if len(content) > 200 else content
            ),
            "title": title,
            "channel": meta.get("channel", "Unknown"),
            "video_id": meta.get("video_id", ""),
            "start_timestamp_formatted": formatted_ts,
            "start_timestamp_seconds": meta.get("start_timestamp_seconds", 0.0),
            "timestamp_url": meta.get("timestamp_url", ""),
            "source_label": f"[{formatted_ts}] {title}",
        }
        sources.append(source_dict)

    return sources


def _has_overlap(text1: str, text2: str, threshold: int = 80) -> bool:
    """
    Check if two text chunks share substantial overlap based on lead anchor probes.
    """
    probe = text1[:100].strip()
    if len(probe) < 20:
        return False
    return probe in text2 or text2[:100].strip() in text1


def deduplicate_chunks(documents: list, similarity_threshold: int = 80) -> list:
    """
    Remove near-duplicate chunks from retrieved results to avoid redundant context.
    """
    if not documents:
        return []

    unique_docs = []
    for doc in documents:
        text = doc.page_content.strip()
        is_duplicate = False

        for unique_doc in unique_docs:
            if _has_overlap(text, unique_doc.page_content.strip(), similarity_threshold):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_docs.append(doc)

    logger.debug("Deduplication: %d -> %d chunks", len(documents), len(unique_docs))
    return unique_docs


def retrieve_and_format(
    retriever,
    query: str,
    deduplicate: bool = True,
) -> tuple[list, list[dict]]:
    """
    Master retrieval function combining query execution, deduplication,
    and structured source formatting.
    """
    documents = retrieve_chunks(retriever, query)
    if deduplicate:
        documents = deduplicate_chunks(documents)
    formatted_sources = format_retrieved_sources(documents)
    return documents, formatted_sources


def build_context_string(documents: list) -> str:
    """
    Combine retrieved Document chunks into a structured context string
    formatted for the grounded generation prompt.
    """
    if not documents:
        return "No relevant context found in the video transcript."

    context_blocks = []
    for doc in documents:
        meta = getattr(doc, "metadata", {})
        ts = meta.get("start_timestamp_formatted", "0:00")
        chunk_id = meta.get("chunk_id", "unknown")
        block = f"[Source: {ts} | Chunk {chunk_id}]\n{doc.page_content.strip()}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def rerank_chunks(
    query: str,
    documents: list,
    reranker,
    top_n: int = 4,
) -> list:
    """
    Take candidate Documents from FAISS and rerank them using a cross-encoder model.
    """
    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

    for doc, score in scored_docs:
        doc.metadata["reranker_score"] = float(score)

    logger.debug("Reranking results for query: '%s'", query[:50])
    for idx, (doc, score) in enumerate(scored_docs):
        chunk_id = doc.metadata.get("chunk_id", "?")
        ts = doc.metadata.get("start_timestamp_formatted", "0:00")
        preview = doc.page_content[:60].replace("\n", " ")
        logger.debug("  Score: %+.3f | Chunk %s | %s | %s...", float(score), chunk_id, ts, preview)

    top_docs = [doc for doc, score in scored_docs[:top_n]]
    logger.info("Reranking: %d candidates → %d selected", len(documents), len(top_docs))
    return top_docs


def build_retriever_with_reranking(
    vectorstore,
    reranker,
    initial_k: int = 20,
    final_k: int = 4,
    search_type: str = "similarity",
):
    """
    Build a two-stage retriever configuration.
    """
    return {
        "retriever": build_retriever(vectorstore, search_type, initial_k),
        "reranker": reranker,
        "initial_k": initial_k,
        "final_k": final_k,
        "mode": "reranked",
    }


def retrieve_and_format_with_reranking(
    query: str,
    retriever_config: dict,
    deduplicate: bool = True,
) -> tuple[list, list[dict]]:
    """
    Master retrieval function for the two-stage reranked pipeline.
    """
    retriever = retriever_config["retriever"]
    reranker = retriever_config["reranker"]
    final_k = retriever_config["final_k"]

    initial_docs = retrieve_chunks(retriever, query)
    if deduplicate:
        initial_docs = deduplicate_chunks(initial_docs)

    reranked_docs = rerank_chunks(query, initial_docs, reranker, top_n=final_k)
    formatted_sources = format_retrieved_sources(reranked_docs)

    logger.info("Two-stage retrieval: %d → %d chunks", len(initial_docs), len(reranked_docs))
    return reranked_docs, formatted_sources


def compare_retrieval_methods(
    query: str,
    vectorstore,
    reranker,
    k: int = 5,
) -> dict:
    """
    Run both plain similarity and reranked retrieval on the same query
    and return a side-by-side comparison dict.
    """
    retriever_a = build_retriever(vectorstore, "similarity", k)
    docs_a, sources_a = retrieve_and_format(retriever_a, query)

    retriever_config = build_retriever_with_reranking(
        vectorstore, reranker, initial_k=k * 4, final_k=k
    )
    docs_b, sources_b = retrieve_and_format_with_reranking(query, retriever_config)

    return {
        "query": query,
        "similarity_results": [
            {
                "chunk_id": s["chunk_id"],
                "timestamp": s["start_timestamp_formatted"],
                "preview": s["text_preview"],
            }
            for s in sources_a
        ],
        "reranked_results": [
            {
                "chunk_id": s["chunk_id"],
                "timestamp": s["start_timestamp_formatted"],
                "reranker_score": docs_b[i].metadata.get("reranker_score", 0)
                if i < len(docs_b)
                else 0,
                "preview": s["text_preview"],
            }
            for i, s in enumerate(sources_b)
        ],
    }
