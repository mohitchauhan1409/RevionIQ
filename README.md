# iq-api

Python FastAPI backend for Revion IQ. Handles AI conversation (Gemini 2.0 Flash), vector search (ChromaDB), voice (ElevenLabs), and real-time WebSocket.

## Setup

### 1. Add API keys to `.env`

```
GEMINI_API_KEY=...        # https://aistudio.google.com/app/apikey
ELEVENLABS_API_KEY=...    # https://elevenlabs.io → Profile → API Keys
```

### 2. Install and run

```bash
cd iq-api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

On first run, ChromaDB indexes all 40 repair cases (~30s).

### 3. Verify

```
GET http://localhost:8000/api/health
```

Should return `{ "status": "ok", "vector_db_cases": 40 }`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/jobs` | All jobs |
| GET | `/api/jobs/:id` | Single job |
| POST | `/api/sessions/start` | Start IQ session |
| POST | `/api/sessions/:id/turn` | Send text turn |
| GET | `/api/sessions/:id/story` | Get C1/C2/C3 |
| POST | `/api/sessions/:id/complete` | Finalize session |
| WS | `/ws/:sessionId` | Real-time voice |

## Deploy (Railway)

1. Push `iq-api/` as its own repo (or connect this folder)
2. Railway → New Project → Deploy from GitHub
3. Set env vars: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
# RevionIQ
