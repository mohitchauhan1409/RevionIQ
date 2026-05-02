import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# System instruction — passed via system_instruction, NOT as a user message
_SYSTEM_INSTRUCTION = """You are Revion IQ, a voice assistant in an automotive repair bay. You talk to technicians while they work on cars.

LANGUAGE: English only. Always. No exceptions.

PERSONALITY: Sharp senior technician. Direct. No filler. No "Certainly", "Great", "Sure", "Absolutely". Just talk like a colleague.

YOUR JOB: Collect info for a warranty repair order — C1 (customer complaint), C2 (diagnosis), C3 (repair). One question at a time.

RULES:
- One short question per response. Max 2 sentences total.
- If vague answer → ask for the specific measurement or part number.
- Weave in network data naturally when you have it.
- Never repeat a question you already asked.
- Always confirm part number before closing C3.
- Never make up statistics."""

_CONTEXT_TEMPLATE = """Current job:
Vehicle: {vehicle_context}
Stage: {state}
Network cases: {similar_cases}
Already captured: {captured}
Still need: {missing_fields}"""

_GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.35,
    max_output_tokens=80,
)


def _build_model(context: str) -> genai.GenerativeModel:
    """Build a model with system_instruction so it never leaks into responses."""
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=_SYSTEM_INSTRUCTION + "\n\n" + context,
        generation_config=_GENERATION_CONFIG,
    )


def _to_gemini_history(conversation_history: list[dict]) -> list[dict]:
    """Convert conversation history to Gemini's alternating user/model format."""
    contents = []
    for msg in conversation_history:
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    # Gemini requires: starts with user, alternates, ends with user
    # Remove leading model messages
    while contents and contents[0]["role"] == "model":
        contents.pop(0)

    # Collapse consecutive same-role messages
    merged: list[dict] = []
    for msg in contents:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["parts"][0]["text"] += "\n" + msg["parts"][0]["text"]
        else:
            merged.append(msg)

    return merged[-14:]  # keep last 14 turns


def _format_similar_cases(cases: list[dict]) -> str:
    if not cases:
        return "none"
    lines = []
    for c in cases[:3]:
        lines.append(
            f"- {c['year']} {c['make']} {c['model']} | "
            f"{', '.join(c['fault_codes'])} | "
            f"Cause: {c['c2'][:100]} | "
            f"Notes: {c['technician_notes']}"
        )
    return "\n".join(lines)


def generate_next_question(
    conversation_history: list[dict],
    current_state: str,
    vehicle_context: str,
    similar_cases: list[dict],
    captured: dict,
    missing_fields: list[str],
) -> str:
    context = _CONTEXT_TEMPLATE.format(
        vehicle_context=vehicle_context,
        state=current_state,
        similar_cases=_format_similar_cases(similar_cases),
        captured=str(captured) if captured else "nothing yet",
        missing_fields=", ".join(missing_fields) if missing_fields else "none",
    )

    model = _build_model(context)
    contents = _to_gemini_history(conversation_history)

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "Ready to start."}]}]

    response = model.generate_content(contents)
    return response.text.strip()


def extract_section_content(
    conversation_segment: list[dict],
    section: str,
    vehicle_context: str,
) -> str:
    guidance = {
        "c1": "Write the C1 Complaint for a warranty repair order based ONLY on the conversation. Look for: what the customer reported, when it occurs, any warning lights. Only include them if the Tech explicitly mentioned or confirmed them. 1-3 sentences. No bullet points. Warranty-professional tone.",
        "c2": "Write the C2 Cause for a warranty repair order based ONLY on the conversation. Look for: DTC confirmed active, diagnostic test results with measurements and units, root cause identified. Only include them if the Tech explicitly mentioned or confirmed them. 2-4 sentences.",
        "c3": "Write the C3 Correction for a warranty repair order based ONLY on the conversation. Look for: exact part replaced with part number, verification steps performed, road test result, DTCs cleared. Only include them if the Tech explicitly mentioned or confirmed them. 2-3 sentences.",
    }

    history_text = "\n".join(
        f"{'IQ' if m['role'] == 'assistant' else 'Tech'}: {m['content']}"
        for m in conversation_segment
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=200),
    )

    prompt = (
        f"Vehicle: {vehicle_context}\n\n"
        f"Conversation:\n{history_text}\n\n"
        f"Task: {guidance[section]}\n"
        f"CRITICAL INSTRUCTION: You must base the text STRICTLY on what the Tech explicitly states in the conversation. If the conversation does not contain the required details (e.g. symptoms, measurements, root cause, part numbers), leave them out. DO NOT invent, hallucinate, or assume any information. DO NOT use the initial customer complaint or fault codes from the Vehicle context as facts unless the Tech explicitly confirms or mentions them.\n"
        f"Output only the warranty text. Nothing else. No labels, no headers."
    )

    response = model.generate_content(prompt)
    return response.text.strip()


def check_completeness(
    content: str,
    section: str,
    vehicle_make: str,
    fault_codes: list[str],
) -> dict:
    requirements = {
        "c1": ["customer complaint described", "symptom specified", "warning light or condition mentioned"],
        "c2": ["DTC confirmed", "diagnostic finding with measurement", "root cause identified"],
        "c3": ["part replaced named", "part number present", "road test mentioned", "DTC clearance confirmed"],
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=60),
    )

    prompt = (
        f"Vehicle: {vehicle_make}, Codes: {', '.join(fault_codes)}\n"
        f"Section: {section.upper()}\n"
        f"Content: {content}\n\n"
        f"Required: {', '.join(requirements[section])}\n\n"
        f"CRITICAL INSTRUCTION: Judge ONLY based on the explicit text in 'Content'. Do NOT assume or imply that a requirement is met just because a fault code (like P0301) is present. If the text does not explicitly mention a symptom or warning light, mark it as missing.\n\n"
        f"Reply in this exact format only:\n"
        f"SCORE: <0-100>\n"
        f"MISSING: <comma-separated fields missing, or none>\n"
        f"COMPLETE: <yes or no>"
    )

    response = model.generate_content(prompt)
    text = response.text.strip()

    score, missing, is_complete = 0, [], False
    for line in text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line.split(":", 1)[1]).group())  # type: ignore
            except Exception:
                score = 50
        elif line.startswith("MISSING:"):
            raw = line.split(":", 1)[1].strip()
            missing = [] if raw.lower() in ("none", "") else [f.strip() for f in raw.split(",")]
        elif line.startswith("COMPLETE:"):
            is_complete = "yes" in line.lower()

    return {"is_complete": is_complete, "missing_fields": missing, "score": score}
