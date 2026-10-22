"""
Generation module.
Handles conversational RAG: conversation memory management,
query rewriting for follow-up questions, grounded prompt assembly,
LLM answer generation, source attribution formatting, and streaming.
Implemented in Phase 6 and Phase 8.
"""

import os
import re
from dotenv import load_dotenv

from app.utils.logger import get_logger
from app.utils.errors import LLMError

logger = get_logger(__name__)

# Support both langchain and langchain_core packaging across environments
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    ConversationBufferMemory = None

try:
    from langchain.schema import AIMessage, HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# Fallback memory class if langchain.memory is not installed
class _SimpleChatMessageHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def clear(self):
        self.messages.clear()


class _FallbackConversationBufferMemory:
    """Lightweight fallback implementing ConversationBufferMemory interface."""

    def __init__(
        self,
        memory_key: str = "chat_history",
        return_messages: bool = True,
        output_key: str = "answer",
    ):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.output_key = output_key
        self.chat_memory = _SimpleChatMessageHistory()

    def clear(self):
        self.chat_memory.clear()

    def load_memory_variables(self, inputs: dict) -> dict:
        if self.return_messages:
            return {self.memory_key: list(self.chat_memory.messages)}
        return {
            self.memory_key: "\n".join(
                f"{m.type}: {m.content}" for m in self.chat_memory.messages
            )
        }

    def save_context(self, inputs: dict, outputs: dict):
        user_input = inputs.get("question") or inputs.get("input", "")
        ai_output = outputs.get(self.output_key) or outputs.get("output", "")
        if user_input:
            self.chat_memory.add_user_message(user_input)
        if ai_output:
            self.chat_memory.add_ai_message(ai_output)


# =====================================================================
# SECTION 1: CONVERSATION MEMORY (Phase 6)
# =====================================================================

"""
About ConversationBufferMemory:
- ConversationBufferMemory stores the sequential list of chat turns (HumanMessage and AIMessage)
  directly in memory during the user's active session.
- memory_key='chat_history': Specifies the key under which conversation history is injected
  into downstream prompt templates or chains.
- return_messages=True: Configures memory to return structured Message objects rather than a single
  flattened string, which allows modern chat models to preserve role metadata.
- output_key='answer': Tells the memory extraction layer which output variable produced by the chain
  represents the AI's final answer to store in history.
- Known limitation: Memory buffer grows monotonically with every turn and does not summarize or prune;
  in very long conversations, it will eventually exceed the LLM's context window.
"""
def create_conversation_memory():
    """
    Create and return a LangChain ConversationBufferMemory instance.
    Stores the full chat history in memory during a session.
    """
    memory_cls = ConversationBufferMemory or _FallbackConversationBufferMemory
    return memory_cls(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )


def format_chat_history(memory) -> str:
    """
    Convert the conversation memory into a formatted string that can be inserted
    into prompts or query rewriters.
    """
    messages = getattr(getattr(memory, "chat_memory", None), "messages", [])
    if not messages:
        return "No previous conversation."

    formatted_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Assistant: {msg.content}")
        else:
            # Handle generic message objects if present
            formatted_lines.append(f"{getattr(msg, 'type', 'User').capitalize()}: {msg.content}")

    return "\n".join(formatted_lines)


def get_chat_history_for_display(memory) -> list[dict]:
    """
    Return the chat history as a list of dictionaries structured for Streamlit chat UI:
    [{"role": "user"|"assistant", "content": "..."}]
    """
    messages = getattr(getattr(memory, "chat_memory", None), "messages", [])
    if not messages:
        return []

    display_list = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            display_list.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            display_list.append({"role": "assistant", "content": msg.content})
        else:
            role = "user" if getattr(msg, "type", "") == "human" else "assistant"
            display_list.append({"role": role, "content": msg.content})

    return display_list


def add_user_message(memory, message: str):
    """
    Add a user message to conversation memory manually.
    """
    memory.chat_memory.add_user_message(message)
    snippet = message[:50] + "..." if len(message) > 50 else message
    logger.debug(f"Added user message to memory: '{snippet}'")


def add_ai_message(memory, message: str):
    """
    Add an AI response to conversation memory manually.
    """
    memory.chat_memory.add_ai_message(message)
    snippet = message[:50] + "..." if len(message) > 50 else message
    logger.debug(f"Added AI message to memory: '{snippet}'")


def clear_memory(memory):
    """
    Clear all messages from the conversation memory.
    Used when the user switches to a new video.
    """
    memory.clear()
    logger.info("Conversation memory cleared.")


# =====================================================================
# SECTION 2: QUERY REWRITING (Phase 6)
# =====================================================================

def needs_rewriting(query: str, chat_history: str) -> bool:
    """
    Determine whether a query needs rewriting before retrieval using heuristics.
    Checks for ambiguous pronouns, conversational continuation phrases, and short questions.
    """
    # If there is no previous conversation, the query is inherently standalone
    if not chat_history or chat_history.strip() == "No previous conversation.":
        return False

    q_lower = query.lower().strip()

    # Condition 1: Query contains pronouns referring to previous context
    pronoun_pattern = r"\b(it|this|that|they|them|its|their|he|she|these|those)\b"
    if re.search(pronoun_pattern, q_lower):
        return True

    # Condition 2: Query starts with continuation phrases
    continuation_starters = (
        "what about",
        "how about",
        "and what",
        "but what",
        "can you",
        "could you",
        "tell me more",
        "elaborate",
        "explain more",
        "why is that",
        "how so",
        "give me",
    )
    if any(q_lower.startswith(phrase) for phrase in continuation_starters):
        return True

    # Condition 3: Query is very short (< 5 words) and conversational history exists
    words = q_lower.split()
    if len(words) < 5:
        return True

    # Condition 4: Query contains "it" or "this" as the subject
    if re.search(r"^(is|why|how|does|can|will)\s+(it|this)\b", q_lower):
        return True

    return False


def rewrite_query(
    query: str,
    chat_history: str,
    llm,
) -> str:
    """
    Use the Groq LLM to rewrite an ambiguous follow-up question into a clear
    standalone question suitable for semantic vector retrieval.
    """
    system_message_text = """You are a query rewriting assistant.
Your job is to rewrite a follow-up question into a clear, 
standalone question that can be understood without any 
conversation history.

Rules:
- Replace all pronouns (it, this, that, they) with the 
  specific concept they refer to
- Make the question self-contained and specific
- Keep it as a question
- Do not answer the question, only rewrite it
- If the question is already standalone, return it unchanged
- Return ONLY the rewritten question, nothing else"""

    human_message_text = f"""Conversation history:
{chat_history}

Follow-up question: {query}

Rewritten standalone question:"""

    try:
        # Construct prompt template with system and human instructions
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_text),
                ("human", human_message_text),
            ]
        )

        # Execute LLM chain
        chain = prompt | llm
        response = chain.invoke({})

        # Extract textual content from ChatResult or Message
        rewritten = getattr(response, "content", str(response))

        # Clean trailing whitespace and enclosing quotes
        rewritten = rewritten.strip().strip('"\'')

        logger.debug(f"Original query: '{query}'")
        logger.info(f"Rewritten query: '{rewritten}'")
        return rewritten

    except Exception as e:
        # Graceful fallback: return original query so the pipeline does not halt
        logger.warning(f"Query rewriting failed, using original: {str(e)}")
        return query


def process_query(
    query: str,
    memory,
    llm,
) -> str:
    """
    Master query processing function.
    Evaluates whether rewriting is needed and rewrites via LLM when appropriate.
    """
    # Step 1: Format existing conversation history
    chat_history = format_chat_history(memory)

    # Step 2: Check if rewriting is needed
    if needs_rewriting(query, chat_history):
        # Step 3: Rewrite ambiguous query
        return rewrite_query(query, chat_history, llm)

    # Step 4: If already standalone, return unchanged
    logger.debug("Query is standalone, no rewriting needed.")
    return query


# =====================================================================
# SECTION 3: GROUNDED GENERATION & ATTRIBUTION (Phase 8)
# =====================================================================

def build_system_prompt() -> str:
    """
    Return the system prompt string that instructs the LLM how to behave.
    This is the most important prompt engineering in the entire project.
    """
    return """You are an intelligent video assistant that helps users understand 
YouTube video content through conversation.

You have access to transcript excerpts from the video. Your job is to 
answer the user's questions based ONLY on the provided transcript context.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the provided transcript context.
2. If the answer is not found in the transcript context (or cannot be found), respond with:
   "I could not find information about this in the video transcript. 
   The video may not cover this topic, or the relevant section may 
   not have been retrieved."
3. Do NOT use your general knowledge or training data to answer questions.
4. Do NOT make up or infer information not present in the transcript.
5. When answering, you may quote short phrases from the transcript directly.
6. Always be conversational and clear in your responses.
7. If the context is partially relevant, use what is available and 
   note what is missing.
8. Keep answers focused and concise unless the user asks for detail.

You are helping the user understand THIS specific video, not the topic 
in general."""


def build_generation_prompt(
    question: str,
    context: str,
    chat_history: str,
) -> list:
    """
    Build the complete list of messages to send to the Groq LLM.
    Returns a list of LangChain message objects.

    Why we use multiple HumanMessages instead of putting everything in one message:
    - Separating context, history, and question into distinct blocks
      makes the structure explicit for the LLM
    - It is easier to debug — you can see exactly what each block contains
    - Some LLM providers handle structured multi-message prompts better
    """
    messages = []

    # Message 1 — SystemMessage: Instructions and grounding constraints
    messages.append(SystemMessage(content=build_system_prompt()))

    # Message 2 — HumanMessage (conversation history block):
    # Only include this message if chat_history is not "No previous conversation."
    # If it is the first message or empty, skip this entirely.
    if chat_history and chat_history.strip() != "No previous conversation.":
        history_content = f"""CONVERSATION HISTORY:
{chat_history}

---"""
        messages.append(HumanMessage(content=history_content))

    # Message 3 — HumanMessage (context block):
    context_content = f"""TRANSCRIPT CONTEXT:
The following are relevant excerpts from the video transcript.
Use these to answer the question.

{context}

---"""
    messages.append(HumanMessage(content=context_content))

    # Message 4 — HumanMessage (question):
    question_content = f"""USER QUESTION:
{question}

Please answer based on the transcript context above.
If the answer is not in the context, say so clearly."""
    messages.append(HumanMessage(content=question_content))

    return messages


def detect_not_found(answer: str) -> bool:
    """
    Detect whether the LLM's answer indicates that the information 
    was not found in the video transcript.

    Check if any of these phrases appear in the answer (case-insensitive):
    - "could not find"
    - "not found in the video"
    - "not in the transcript"
    - "video may not cover"
    - "not covered in"
    - "no information"
    - "cannot find"
    - "does not appear"
    - "not mentioned"
    - "not discussed"

    Return True if any phrase is found, False otherwise.
    This is used by the UI to show a special "not found" indicator
    instead of treating the response as a normal answer.
    """
    phrases = [
        "could not find",
        "not found in the video",
        "not in the transcript",
        "video may not cover",
        "not covered in",
        "no information",
        "cannot find",
        "does not appear",
        "not mentioned",
        "not discussed",
    ]
    ans_lower = answer.lower()
    return any(phrase in ans_lower for phrase in phrases)


def format_sources_for_display(formatted_sources: list[dict]) -> str:
    """
    Convert the list of formatted source dicts into a clean markdown 
    string for display in the Streamlit UI.

    For each source dict, create a line like:
    📍 [{start_timestamp_formatted}]({timestamp_url}) — {text_preview}

    If formatted_sources is empty:
    Return "No specific sources found for this answer."

    If a source has no timestamp_url, just show the timestamp without a link:
    📍 [{start_timestamp_formatted}] — {text_preview}
    """
    if not formatted_sources:
        return "No specific sources found for this answer."

    lines = []
    for source in formatted_sources:
        timestamp = source.get("start_timestamp_formatted", "0:00")
        preview = source.get("text_preview", source.get("text", "")[:200])
        url = source.get("timestamp_url", "")

        # Render clickable timestamp markdown hyperlink if URL is available
        if url:
            lines.append(f"📍 [{timestamp}]({url}) — {preview}")
        else:
            lines.append(f"📍 [{timestamp}] — {preview}")

    return "**Sources from the video:**\n\n" + "\n\n".join(lines)


def generate_answer(
    query: str,
    retriever_or_config,
    memory: ConversationBufferMemory,
    llm,
    use_reranking: bool = False,
) -> tuple[str, list[dict], bool]:
    """
    The complete answer generation function.
    This replaces generate_answer_placeholder entirely.
    This is the main function the Streamlit UI calls on every user message.

    Returns a tuple of:
    - answer: str (the LLM generated answer)
    - formatted_sources: list[dict] (source chunks with timestamps)
    - not_found: bool (True if answer was not in the video)
    """
    try:
        # Step 1 — Process the query:
        # Rewrites follow-up questions using conversation history into standalone queries
        retrieval_query = process_query(query, memory, llm)
        logger.info(f"Retrieval query: '{retrieval_query}'")

        # Step 2 — Retrieve relevant chunks:
        # Uses single-stage FAISS similarity or two-stage Cross-Encoder reranking
        from app.retrieval.retriever import (
            build_context_string,
            retrieve_and_format,
            retrieve_and_format_with_reranking,
        )

        if use_reranking:
            documents, formatted_sources = retrieve_and_format_with_reranking(
                retrieval_query, retriever_or_config
            )
        else:
            documents, formatted_sources = retrieve_and_format(
                retriever_or_config, retrieval_query
            )

        # Step 3 — Build context string:
        # Assembles tagged, timestamped transcript snippets for the prompt
        context = build_context_string(documents)

        # Step 4 — Get conversation history:
        # Formats active session dialog history for contextual continuity
        chat_history = format_chat_history(memory)

        # Step 5 — Build prompt messages:
        # Assembles system prompt, conversation history, retrieved context, and question
        messages = build_generation_prompt(query, context, chat_history)

        # Step 6 — Call the Groq LLM:
        # Executes inference via Groq LLaMA 3.3 70B
        response = llm.invoke(messages)
        answer = response.content.strip()
        logger.info(f"Answer generated ({len(answer)} characters)")

        # Step 7 — Detect not found:
        # Checks if LLM indicated the requested information is absent from transcript
        not_found = detect_not_found(answer)
        if not_found:
            logger.info("Note: Answer indicates information not found in video.")

        # Step 8 — Update conversation memory:
        # Appends current turn to conversation buffer memory
        add_user_message(memory, query)
        add_ai_message(memory, answer)

        # Step 9 — Return:
        return (answer, formatted_sources, not_found)

    except Exception as e:
        logger.error("Answer generation error: %s", str(e))
        raise LLMError(f"Answer generation failed: {str(e)}")


def generate_answer_with_sources(
    query: str,
    retriever_or_config,
    memory: ConversationBufferMemory,
    llm,
    use_reranking: bool = False,
) -> dict:
    """
    Convenience wrapper around generate_answer() that returns 
    everything as a single structured dict.
    This is the cleanest interface for the Streamlit UI.
    """
    answer, formatted_sources, not_found = generate_answer(
        query=query,
        retriever_or_config=retriever_or_config,
        memory=memory,
        llm=llm,
        use_reranking=use_reranking,
    )

    return {
        "answer": answer,
        "sources": formatted_sources,
        "sources_markdown": format_sources_for_display(formatted_sources),
        "not_found": not_found,
        "query": query,
        "num_sources": len(formatted_sources),
    }


def stream_answer(
    query: str,
    retriever_or_config,
    memory: ConversationBufferMemory,
    llm,
    use_reranking: bool = False,
):
    """
    Generator function that streams the answer token by token.
    Used for a better UX in Streamlit where the answer appears 
    word by word instead of all at once.

    Why a generator:
    - 'yield' makes this a Python generator
    - Streamlit's st.write_stream() can consume this directly
    - This gives users immediate feedback instead of waiting
      for the full response
    """
    from app.retrieval.retriever import (
        build_context_string,
        retrieve_and_format,
        retrieve_and_format_with_reranking,
    )

    # Step 1: Process and rewrite query if follow-up
    retrieval_query = process_query(query, memory, llm)

    # Step 2: Retrieve relevant chunks
    if use_reranking:
        documents, formatted_sources = retrieve_and_format_with_reranking(
            retrieval_query, retriever_or_config
        )
    else:
        documents, formatted_sources = retrieve_and_format(
            retriever_or_config, retrieval_query
        )

    # Step 3: Build context string
    context = build_context_string(documents)

    # Step 4: Build chat history
    chat_history = format_chat_history(memory)

    # Step 5: Build messages
    messages = build_generation_prompt(query, context, chat_history)

    # Step 6: Stream the response using llm.stream(messages)
    full_answer = ""
    for chunk in llm.stream(messages):
        token = getattr(chunk, "content", str(chunk))
        full_answer += token
        yield token  # yield each token as it arrives

    # Step 7: After streaming completes, update memory
    add_user_message(memory, query)
    add_ai_message(memory, full_answer)


# =====================================================================
# SECTION 4: BACKWARDS COMPATIBILITY
# =====================================================================

def generate_answer_placeholder(
    query: str,
    retriever,
    memory,
    llm,
    use_reranking: bool = False,
    retriever_config: dict = None,
) -> tuple[str, list[dict]]:
    """
    Backwards-compatible wrapper calling generate_answer().
    Returns (answer, formatted_sources). Kept so earlier phase tests continue to pass.
    """
    config = retriever_config if use_reranking else retriever
    answer, sources, _ = generate_answer(
        query=query,
        retriever_or_config=config,
        memory=memory,
        llm=llm,
        use_reranking=use_reranking,
    )
    return answer, sources
