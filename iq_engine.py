from enum import Enum
from dataclasses import dataclass
from conversation import ConversationHistory
from gemini_client import generate_next_question, extract_section_content, check_completeness
from vector_store import get_vector_store

# Minimum user turns required before writing or advancing each section
_MIN_TURNS = {"c1": 2, "c2": 3, "c3": 2}
# Completeness score required to advance
_ADVANCE_THRESHOLD = 82


class IQState(str, Enum):
    GREETING = "GREETING"
    C1_GATHERING = "C1_GATHERING"
    C2_BRIEFING = "C2_BRIEFING"
    C2_GATHERING = "C2_GATHERING"
    C3_GATHERING = "C3_GATHERING"
    C3_COMPLETE = "C3_COMPLETE"


@dataclass
class TurnResult:
    ai_response: str
    c1: str
    c2: str
    c3: str
    state: str
    network_insight: dict | None
    is_complete: bool
    missing_fields: list[str]
    section_scores: dict


class IQEngine:
    def __init__(self, job: dict):
        self.job = job
        self.vehicle = job["vehicle"]
        self.fault_codes = job.get("fault_codes", [])
        self.state = IQState.GREETING
        self.history = ConversationHistory()
        self.c1 = job.get("c1", "")
        self.c2 = job.get("c2", "")
        self.c3 = job.get("c3", "")
        self.section_scores = {"c1": 0, "c2": 0, "c3": 0}
        self._similar_cases: list[dict] = []
        self._network_insight: dict | None = None
        self._c1_start = 0
        self._c2_start = 0
        self._c3_start = 0

        store = get_vector_store()
        self._similar_cases = store.search_similar_cases(
            fault_codes=self.fault_codes,
            vehicle_make=self.vehicle["make"],
            vehicle_model=self.vehicle["model"],
            symptom=job.get("customer_complaint", ""),
            n=5,
        )

        if self._similar_cases:
            top = self._similar_cases[0]
            self._network_insight = {
                "text": (
                    f"{round(top['similarity_score'] * 100)}% match — "
                    f"{top['year']} {top['make']} {top['model']}: {top['technician_notes']}"
                ),
                "common_mistakes": top.get("common_mistakes", []),
                "similar_count": len(self._similar_cases),
            }

    def _user_turns_since(self, from_index: int) -> int:
        return sum(1 for m in self.history.get_segment(from_index) if m["role"] == "user")

    @property
    def _vehicle_context(self) -> str:
        v = self.vehicle
        return (
            f"{v['year']} {v['make']} {v['model']} ({v.get('engine', 'N/A')}) | "
            f"{v.get('mileage', 0):,} miles | "
            f"Fault codes: {', '.join(self.fault_codes)} | "
            f"Customer complaint: {self.job.get('customer_complaint', '')}"
        )

    @property
    def _basic_vehicle_context(self) -> str:
        v = self.vehicle
        return (
            f"{v['year']} {v['make']} {v['model']} ({v.get('engine', 'N/A')}) | "
            f"{v.get('mileage', 0):,} miles"
        )

    def _greeting_message(self) -> str:
        v = self.vehicle
        return (
            f"Hey — I've got the {v['year']} {v['make']} {v['model']} "
            f"pulled up, {', '.join(self.fault_codes)} confirmed. "
            f"What's the customer telling you about it?"
        )

    def _c2_briefing_message(self) -> str:
        if self._network_insight:
            return (
                f"Alright, C1's locked in. "
                f"Network data on this one: {self._network_insight['text']}. "
                f"What are your diagnostic findings?"
            )
        return "Alright, C1's locked in. Walk me through your diagnostic findings."

    def _try_extract(self, section: str, start: int) -> bool:
        """Extract section content only when we have enough turns. Returns True if extracted."""
        if self._user_turns_since(start) < _MIN_TURNS[section]:
            return False
        segment = self.history.get_segment(start)
        content = extract_section_content(segment, section, self._basic_vehicle_context)
        if content and len(content) > 30:
            setattr(self, section, content)
            return True
        return False

    def _check_advance(self, section: str) -> tuple[bool, list[str]]:
        """Check completeness and return (should_advance, missing_fields)."""
        content = getattr(self, section)
        if not content:
            return False, [f"{section} content not yet written"]
        result = check_completeness(content, section, self.vehicle["make"], self.fault_codes)
        self.section_scores[section] = result["score"]
        return result["score"] >= _ADVANCE_THRESHOLD, result.get("missing_fields", [])

    def process_turn(self, technician_input: str) -> TurnResult:
        self.history.add("user", technician_input)

        ai_response = ""
        network_insight = None
        missing_fields: list[str] = []



        # ── GREETING → C1 ────────────────────────────────────────────────────
        if self.state == IQState.GREETING:
            self._c1_start = len(self.history) - 1
            self.state = IQState.C1_GATHERING
            ai_response = generate_next_question(
                self.history.get_all(), self.state.value, self._vehicle_context,
                self._similar_cases, {},
                ["what exactly is the customer experiencing", "any warning lights", "when does it happen"],
            )

        # ── C1 GATHERING ─────────────────────────────────────────────────────
        elif self.state == IQState.C1_GATHERING:
            self._try_extract("c1", self._c1_start)
            advance, missing_fields = self._check_advance("c1")

            if advance:
                self.state = IQState.C2_BRIEFING
                ai_response = self._c2_briefing_message()
                network_insight = self._network_insight
                self._c2_start = len(self.history)
            else:
                ai_response = generate_next_question(
                    self.history.get_all(), self.state.value, self._vehicle_context,
                    self._similar_cases, {"c1": self.c1}, missing_fields,
                )

        # ── C2 BRIEFING → C2 ─────────────────────────────────────────────────
        elif self.state == IQState.C2_BRIEFING:
            self.state = IQState.C2_GATHERING
            self._c2_start = len(self.history) - 1
            ai_response = generate_next_question(
                self.history.get_all(), self.state.value, self._vehicle_context,
                self._similar_cases, {"c1": self.c1},
                ["DTC confirmed", "which component failed", "measurement that proves it"],
            )

        # ── C2 GATHERING ─────────────────────────────────────────────────────
        elif self.state == IQState.C2_GATHERING:
            self._try_extract("c2", self._c2_start)
            advance, missing_fields = self._check_advance("c2")

            if advance:
                self.state = IQState.C3_GATHERING
                self._c3_start = len(self.history)
                ai_response = generate_next_question(
                    self.history.get_all(), self.state.value, self._vehicle_context,
                    self._similar_cases, {"c1": self.c1, "c2": self.c2},
                    ["what part was replaced", "part number", "road test result", "DTCs cleared"],
                )
            else:
                ai_response = generate_next_question(
                    self.history.get_all(), self.state.value, self._vehicle_context,
                    self._similar_cases, {"c1": self.c1, "c2": self.c2}, missing_fields,
                )

        # ── C3 GATHERING ─────────────────────────────────────────────────────
        elif self.state == IQState.C3_GATHERING:
            self._try_extract("c3", self._c3_start)
            advance, missing_fields = self._check_advance("c3")

            if advance:
                self.state = IQState.C3_COMPLETE
                ai_response = "That's everything. C1, C2, C3 are all warranty-ready. Ready to submit?"
            else:
                ai_response = generate_next_question(
                    self.history.get_all(), self.state.value, self._vehicle_context,
                    self._similar_cases,
                    {"c1": self.c1, "c2": self.c2, "c3": self.c3}, missing_fields,
                )

        elif self.state == IQState.C3_COMPLETE:
            ai_response = "Story's already complete — go ahead and submit to the DMS."

        self.history.add("assistant", ai_response)

        return TurnResult(
            ai_response=ai_response,
            c1=self.c1,
            c2=self.c2,
            c3=self.c3,
            state=self.state.value,
            network_insight=network_insight,
            is_complete=self.state == IQState.C3_COMPLETE,
            missing_fields=missing_fields,
            section_scores=dict(self.section_scores),
        )

    def get_story_completeness(self) -> dict:
        scores = {}
        total = 0
        for section in ["c1", "c2", "c3"]:
            content = getattr(self, section)
            if content:
                result = check_completeness(content, section, self.vehicle["make"], self.fault_codes)
                scores[section] = result["score"]
                total += result["score"]
            else:
                scores[section] = 0
        return {
            "scores": scores,
            "overall": round(total / 3),
            "is_warranty_ready": all(s >= _ADVANCE_THRESHOLD for s in scores.values()),
        }
