"""
Transcript ingestion module.
Handles extracting the transcript from a YouTube video,
fetching video metadata (title, channel, duration),
and cleaning the raw transcript text before processing.
"""

from urllib.parse import parse_qs, urlparse
import re
from dotenv import load_dotenv

from app.utils.errors import (
    InvalidURLError,
    TranscriptNotAvailableError,
    VideoUnavailableError,
)
from app.utils.logger import get_logger

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from youtube_transcript_api import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )
except ImportError:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception
    VideoUnavailable = Exception

# Central logger instance
logger = get_logger(__name__)

# Load environment variables if configured
load_dotenv()


def extract_video_id(url: str) -> str:
    """
    Parse a YouTube URL and return just the video ID string.

    Supports:
    - Standard: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - Shortened: https://youtu.be/dQw4w9WgXcQ
    - Embed: https://www.youtube.com/embed/dQw4w9WgXcQ
    - Timestamps/params: ?t=120s, &list=PLxxx
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError(str(url))

    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    path = parsed.path
    video_id = None

    if "youtube.com" in hostname:
        if path == "/watch":
            query_params = parse_qs(parsed.query)
            v_param = query_params.get("v")
            if v_param and len(v_param) > 0 and v_param[0]:
                video_id = v_param[0]
        elif path.startswith("/embed/"):
            path_parts = path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[1]:
                video_id = path_parts[1]
        elif path.startswith("/v/"):
            path_parts = path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[1]:
                video_id = path_parts[1]

    elif "youtu.be" in hostname:
        path_parts = path.strip("/").split("/")
        if len(path_parts) >= 1 and path_parts[0]:
            video_id = path_parts[0]

    if not video_id:
        raise InvalidURLError(url)

    return video_id


def fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch video title, channel, and duration using yt-dlp.
    yt-dlp is more reliable than pytube for YouTube metadata.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "video_id": video_id,
                "title": info.get("title", "Unknown Title"),
                "channel": info.get(
                    "uploader", info.get("channel", "Unknown Channel")
                ),
                "duration_seconds": info.get("duration", 0),
                "url": url,
            }
    except Exception as e:
        logger.warning("yt-dlp metadata fetch failed: %s", str(e))
        return {
            "video_id": video_id,
            "title": "Unknown Title",
            "channel": "Unknown Channel",
            "duration_seconds": 0,
            "url": url,
        }


def fetch_transcript(video_id: str) -> list[dict]:
    """
    Fetch raw transcript segments for a YouTube video using YouTubeTranscriptApi.
    """
    if YouTubeTranscriptApi is None:
        raise TranscriptNotAvailableError(
            video_id, "youtube-transcript-api is not installed."
        )

    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            api = YouTubeTranscriptApi()
            raw = api.fetch(video_id)
            transcript = [
                {
                    "text": getattr(s, "text", s.get("text", "") if isinstance(s, dict) else str(s)),
                    "start": getattr(s, "start", s.get("start", 0.0) if isinstance(s, dict) else 0.0),
                    "duration": getattr(s, "duration", s.get("duration", 0.0) if isinstance(s, dict) else 0.0),
                }
                for s in raw
            ]
        logger.info("Fetched %d transcript segments for video %s", len(transcript), video_id)
        return transcript
    except TranscriptsDisabled:
        raise TranscriptNotAvailableError(video_id, "Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise TranscriptNotAvailableError(video_id, "No transcript found for this video.")
    except VideoUnavailable:
        raise VideoUnavailableError(video_id)
    except Exception as e:
        raise TranscriptNotAvailableError(video_id, str(e))


def clean_transcript_text(text: str) -> str:
    """
    Clean a transcript text segment before it is embedded.
    Strips fillers, collapses extra whitespaces and newlines.
    """
    if not text:
        return ""

    cleaned = text.strip()
    original_stripped = cleaned

    cleaned = re.sub(
        r"\b(um|uh|hmm|mhm|uh-huh|you know)\b\s*,?", "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(r",\s*\blike\b\s*,", ",", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\blike\b\s*,?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r",?\s*\blike\b\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned)
    cleaned = cleaned.strip(" ,")

    if not cleaned:
        return original_stripped

    return cleaned


def build_transcript_documents(url: str) -> list[Document]:
    """
    Parse a YouTube URL, fetch metadata & transcripts, clean, and return LangChain Documents.
    """
    video_id = extract_video_id(url)
    metadata = fetch_video_metadata(video_id)
    segments = fetch_transcript(video_id)

    documents = []
    for segment in segments:
        cleaned_text = clean_transcript_text(segment.get("text", ""))
        if not cleaned_text or len(cleaned_text) < 5:
            continue

        doc = Document(
            page_content=cleaned_text,
            metadata={
                "video_id": video_id,
                "title": metadata["title"],
                "channel": metadata["channel"],
                "duration_seconds": metadata["duration_seconds"],
                "url": url,
                "start": segment.get("start", 0.0),
                "duration": segment.get("duration", 0.0),
                "source": f"YouTube: {metadata['title']}",
            },
        )
        documents.append(doc)

    logger.info("Built %d documents from transcript", len(documents))
    return documents
