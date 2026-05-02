import asyncio
import base64
import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from elevenlabs_client import speech_to_text, text_to_speech
from iq_engine import IQEngine
from vector_store import get_vector_store

load_dotenv()

# Non-ASCII script ranges: CJK, Korean, Arabic, Devanagari, etc.
_NON_LATIN = re.compile(r'[ऀ-ॿ؀-ۿ一-鿿가-힯぀-ヿ]')
# Sound event tags that Scribe adds: (music), (applause), [NOISE], etc.
_SOUND_TAG = re.compile(r'[\(\[\{][^\)\]\}]{1,40}[\)\]\}]')


def _is_valid_transcript(text: str) -> bool:
    """Return False if transcript looks like noise, music, or non-English speech."""
    # Contains non-Latin scripts (Korean, Chinese, Arabic, Hindi…)
    if _NON_LATIN.search(text):
        return False
    # Entire text is just a sound tag like "(music)" or "[background noise]"
    cleaned = _SOUND_TAG.sub("", text).strip()
    if not cleaned:
        return False
    # Fewer than 2 real words after stripping punctuation
    words = [w for w in re.split(r'\s+', cleaned) if re.search(r'[a-zA-Z0-9]', w)]
    if len(words) < 2:
        return False
    return True


app = FastAPI(title="Revion IQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS_PATH = Path(__file__).parent / "data" / "jobs.json"

_sessions: dict[str, IQEngine] = {}
_jobs: dict[str, dict] = {}


def _load_jobs():
    for path in [JOBS_PATH]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return {j["id"]: j for j in data["jobs"]}
    return {}


def _warm_store():
    store = get_vector_store()
    print(f"[iq-api] ChromaDB ready — {store.case_count} repair cases indexed")


@app.on_event("startup")
async def startup():
    global _jobs
    _jobs = _load_jobs()
    print(f"[iq-api] Loaded {len(_jobs)} jobs")
    # Warm vector store in background so the port opens immediately
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(loop.run_in_executor(None, _warm_store))


# ─── Models ───────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    job_id: str
    technician_id: str = "tech_001"


class TurnRequest(BaseModel):
    input_text: str


# ─── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    loop = asyncio.get_event_loop()
    store = await loop.run_in_executor(None, get_vector_store)
    return {
        "status": "ok",
        "vector_db_cases": store.case_count,
        "model": "gemini-2.0-flash",
        "sessions_active": len(_sessions),
    }


@app.get("/api/jobs")
async def get_jobs():
    return {"jobs": list(_jobs.values())}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@app.post("/api/sessions/start")
async def start_session(body: StartSessionRequest):
    job = _jobs.get(body.job_id)
    if not job:
        raise HTTPException(404, f"Job {body.job_id} not found")

    session_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    engine = await loop.run_in_executor(None, lambda: IQEngine(job))
    _sessions[session_id] = engine

    greeting = engine._greeting_message()
    engine.history.add("assistant", greeting)

    similar_preview = [
        {
            "make": c["make"],
            "model": c["model"],
            "year": c["year"],
            "fault_codes": c["fault_codes"],
            "technician_notes": c["technician_notes"],
            "similarity_score": c["similarity_score"],
        }
        for c in engine._similar_cases[:3]
    ]

    return {
        "session_id": session_id,
        "greeting_message": greeting,
        "similar_cases_preview": similar_preview,
        "network_insight": engine._network_insight,
    }


@app.post("/api/sessions/{session_id}/turn")
async def session_turn(session_id: str, body: TurnRequest):
    engine = _sessions.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    result = engine.process_turn(body.input_text)
    return {
        "ai_response": result.ai_response,
        "c1": result.c1,
        "c2": result.c2,
        "c3": result.c3,
        "state": result.state,
        "network_insight": result.network_insight,
        "is_complete": result.is_complete,
        "missing_fields": result.missing_fields,
        "section_scores": result.section_scores,
        "chat_history": engine.history.get_all(),
    }


@app.get("/api/sessions/{session_id}/story")
async def get_story(session_id: str):
    engine = _sessions.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    completeness = engine.get_story_completeness()
    return {
        "c1": engine.c1,
        "c2": engine.c2,
        "c3": engine.c3,
        "completeness_score": completeness["overall"],
        "section_scores": completeness["scores"],
        "is_warranty_ready": completeness["is_warranty_ready"],
        "state": engine.state.value,
    }


@app.post("/api/sessions/{session_id}/complete")
async def complete_session(session_id: str):
    engine = _sessions.get(session_id)
    if not engine:
        raise HTTPException(404, "Session not found")

    completeness = engine.get_story_completeness()

    job = engine.job
    job["c1"] = engine.c1
    job["c2"] = engine.c2
    job["c3"] = engine.c3
    job["story_complete"] = completeness["is_warranty_ready"]
    job["status"] = "complete" if completeness["is_warranty_ready"] else job["status"]

    if job["id"] in _jobs:
        _jobs[job["id"]] = job
        
        # Save to jobs.json
        if JOBS_PATH.exists():
            with open(JOBS_PATH, "r") as f:
                data = json.load(f)
            
            for i, j in enumerate(data["jobs"]):
                if j["id"] == job["id"]:
                    data["jobs"][i] = job
                    break
            else:
                data["jobs"].append(job)
                
            with open(JOBS_PATH, "w") as f:
                json.dump(data, f, indent=2)

    return {
        "final_story": {"c1": engine.c1, "c2": engine.c2, "c3": engine.c3},
        "warranty_ready_score": completeness["overall"],
        "is_warranty_ready": completeness["is_warranty_ready"],
    }


# ─── WebSocket ────────────────────────────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    engine = _sessions.get(session_id)
    if not engine:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return

    audio_chunks: list[bytes] = []

    # Auto-play greeting on connect
    greeting_history = engine.history.get_all()
    if greeting_history and greeting_history[0]["role"] == "assistant":
        greeting_text = greeting_history[0]["content"]
        try:
            audio_bytes = await text_to_speech(greeting_text)
            audio_b64 = base64.b64encode(audio_bytes).decode()
            await websocket.send_json({"type": "greeting_audio", "data": audio_b64})
        except Exception:
            await websocket.send_json({"type": "tts_fallback", "text": greeting_text})

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                chunk = base64.b64decode(msg["data"])
                audio_chunks.append(chunk)

            elif msg_type == "audio_end":
                if not audio_chunks:
                    await websocket.send_json({"type": "error", "message": "No audio received"})
                    audio_chunks = []
                    continue

                full_audio = b"".join(audio_chunks)
                audio_chunks = []

                # Gate: reject if audio is too small to be real speech (< 4 KB)
                if len(full_audio) < 4096:
                    await websocket.send_json({"type": "noise_rejected", "message": "Too short — speak clearly and try again."})
                    continue

                # STT
                try:
                    transcript = await speech_to_text(full_audio)
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"STT failed: {e}"})
                    continue

                # Validate transcript — reject noise, music, non-English
                transcript = transcript.strip()
                if not transcript or not _is_valid_transcript(transcript):
                    await websocket.send_json({"type": "noise_rejected", "message": "Didn't catch that — speak clearly and try again."})
                    continue

                await websocket.send_json({"type": "transcript", "text": transcript})

                # AI turn
                result = engine.process_turn(transcript)

                await websocket.send_json({
                    "type": "ai_response",
                    "text": result.ai_response,
                    "c1": result.c1,
                    "c2": result.c2,
                    "c3": result.c3,
                    "state": result.state,
                    "network_insight": result.network_insight,
                    "is_complete": result.is_complete,
                    "missing_fields": result.missing_fields,
                    "section_scores": result.section_scores,
                })

                if result.c1:
                    await websocket.send_json({"type": "writing", "section": "c1", "text": result.c1})
                if result.c2:
                    await websocket.send_json({"type": "writing", "section": "c2", "text": result.c2})
                if result.c3:
                    await websocket.send_json({"type": "writing", "section": "c3", "text": result.c3})

                # TTS
                try:
                    audio_bytes = await text_to_speech(result.ai_response)
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    await websocket.send_json({"type": "audio", "data": audio_b64})
                except Exception:
                    # ElevenLabs failed — frontend falls back to browser TTS
                    await websocket.send_json({"type": "tts_fallback", "text": result.ai_response})

            elif msg_type == "text_input":
                text = msg.get("text", "").strip()
                if not text:
                    continue

                result = engine.process_turn(text)

                await websocket.send_json({
                    "type": "ai_response",
                    "text": result.ai_response,
                    "c1": result.c1,
                    "c2": result.c2,
                    "c3": result.c3,
                    "state": result.state,
                    "network_insight": result.network_insight,
                    "is_complete": result.is_complete,
                    "missing_fields": result.missing_fields,
                    "section_scores": result.section_scores,
                })

                if result.c1:
                    await websocket.send_json({"type": "writing", "section": "c1", "text": result.c1})
                if result.c2:
                    await websocket.send_json({"type": "writing", "section": "c2", "text": result.c2})
                if result.c3:
                    await websocket.send_json({"type": "writing", "section": "c3", "text": result.c3})

                try:
                    audio_bytes = await text_to_speech(result.ai_response)
                    audio_b64 = base64.b64encode(audio_bytes).decode()
                    await websocket.send_json({"type": "audio", "data": audio_b64})
                except Exception:
                    await websocket.send_json({"type": "tts_fallback", "text": result.ai_response})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
