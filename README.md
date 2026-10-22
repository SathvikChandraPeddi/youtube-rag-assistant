# YouTube RAG Assistant

YouTube RAG Assistant is a production-grade Retrieval-Augmented Generation (RAG) web application that enables conversational interactions with YouTube video content. By combining automated transcript ingestion, semantic text chunking, local vector embeddings, and ultra-fast inference via Groq's LLaMA 3.3 70B model, the application delivers accurate, hallucination-resistant answers with precise timestamp citations deep-linking back to the source video.

## Tech Stack
- **LLM:** LLaMA 3.3 70B via Groq API
- **Embeddings:** all-MiniLM-L6-v2 (HuggingFace, local)
- **Vector Store:** FAISS
- **Framework:** LangChain
- **UI:** Streamlit
- **Transcript:** YouTube Transcript API

## Architecture
The application is structured around two core pipelines:
- **Indexing pipeline:** YouTube URL → Transcript Extraction & Cleaning → Timestamped Chunks → Local HuggingFace Embeddings → FAISS Vector Store Index
- **Query pipeline:** User Question → Conversational Query Rewrite → Semantic Retrieval (Top-k) → Grounded Prompt Assembly → Groq LLM Inference → Timestamp-Cited Answer

## Setup
Follow these steps to set up and run the project locally:

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd youtube-rag-assistant
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
6. Add your `GROQ_API_KEY` in `.env` (get it free at [console.groq.com](https://console.groq.com)).
7. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
Coming soon.

## Evaluation
Coming soon.
