from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MAIN_MODEL = "claude-haiku-4-5"
HAIKU_MODEL = "claude-haiku-4-5"            # compression + fact-check
MAX_HISTORY_CHARS = 6000                    # compress when raw history exceeds this
MAX_TOKENS_MAIN = 1024
MAX_TOKENS_HAIKU = 512
MAX_TOKENS_FACTCHECK = 256

DRIFT_TAG = "[DRIFT WARNING]"
CORRECTION_TAG = "[CORRECTION REQUESTED]"

@dataclass
class SessionState:
    original_goal: str = ""
    established_facts: list[str] = field(default_factory=list)
    active_constraints: list[str] = field(default_factory=list)
    current_thread: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "original_goal": self.original_goal,
                "established_facts": self.established_facts,
                "active_constraints": self.active_constraints,
                "current_thread": self.current_thread,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        try:
            data = json.loads(raw)
            return cls(
                original_goal=data.get("original_goal", ""),
                established_facts=data.get("established_facts", []),
                active_constraints=data.get("active_constraints", []),
                current_thread=data.get("current_thread", ""),
            )
        except json.JSONDecodeError:
            return cls()

class TVASession:
    """
    A hallucination-reducing wrapper around the Anthropic client.

    Parameters
    ----------
    goal : str
        The original user goal that anchors the whole session.
    system_prompt : str, optional
        Extra system instructions merged on top of the state block.
    api_key : str, optional
        Anthropic API key (falls back to ANTHROPIC_API_KEY env var).
    """

    def __init__(
        self,
        goal: str,
        system_prompt: str = "",
        api_key: Optional[str] = None,
    ) -> None:
        self._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._extra_system = system_prompt
        self._history: list[dict] = []          # raw message dicts
        self._compressed_prefix: str = ""       # summary of compressed history
        self._turn_count: int = 0
        self._last_was_flagged: bool = False

        self.state = SessionState(original_goal=goal)


    def chat(
        self,
        user_message: str,
        on_token: Optional[callable] = None,
    ) -> tuple[str, dict]:
        self._turn_count += 1

        # 4. Anchor injection — build system prompt (zero extra calls)
        system = self._build_system_prompt()

        # Append user turn to history
        self._history.append({"role": "user", "content": user_message})
        assistant_reply = self._call_main(system, on_token=on_token)

        # Append assistant turn to history
        self._history.append({"role": "assistant", "content": assistant_reply})

        # 3. Fact grounding check
        flagged, drift_detail = self._fact_grounding_check(assistant_reply)
        self._last_was_flagged = flagged

        corrected = False
        if flagged:
            assistant_reply = self._correction_pass(assistant_reply)
            # Replace last history entry with corrected reply
            self._history[-1]["content"] = assistant_reply
            corrected = True

        # 2. Update structured state
        self._update_state(user_message, assistant_reply)

        # 1. Context compression — trigger when raw history actually grows too large
        history_chars = sum(len(m["content"]) for m in self._history)
        if history_chars > MAX_HISTORY_CHARS:
            self._compress_history()

        meta = {
            "flagged": flagged,
            "drift_detail": drift_detail,
            "corrected": corrected,
            "turn_count": self._turn_count,
        }
        return assistant_reply, meta


    def _build_system_prompt(self) -> str:
        """Assemble the full system prompt injected on every call."""
        anchor = textwrap.dedent(f"""
            ┌─ SESSION STATE ───────────────────────────────────────────────
            │ ORIGINAL GOAL : {self.state.original_goal}
            │
            │ ESTABLISHED FACTS:
            {self._bullet(self.state.established_facts)}
            │
            │ ACTIVE CONSTRAINTS:
            {self._bullet(self.state.active_constraints)}
            │
            │ CURRENT THREAD : {self.state.current_thread}
            └───────────────────────────────────────────────────────────────

            FULL STATE JSON (authoritative source of truth):
            {self.state.to_json()}
        """).strip()

        compressed_section = ""
        if self._compressed_prefix:
            compressed_section = (
                "\n\n── COMPRESSED HISTORY SUMMARY ──\n"
                + self._compressed_prefix
                + "\n── END SUMMARY ──"
            )

        base_instruction = (
            "You are a precise, grounded assistant. "
            "Never invent facts not present in the conversation or state above. "
            "If unsure, say so explicitly. "
            "IMPORTANT: Your replies must be plain conversational text only. "
            "Never output JSON, state blocks, or structured data — "
            "the state above is read-only context for your awareness, not something you should reproduce or update in your replies."
        )

        parts = [anchor, compressed_section, base_instruction]
        if self._extra_system:
            parts.append(self._extra_system)

        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _bullet(items: list[str]) -> str:
        if not items:
            return "│   (none)"
        return "\n".join(f"│   • {item}" for item in items)

    def _call_main(self, system: str, on_token: Optional[callable] = None) -> str:
        messages = self._build_messages()
        if on_token is not None:
            with self._client.messages.stream(
                model=MAIN_MODEL,
                max_tokens=MAX_TOKENS_MAIN,
                system=system,
                messages=messages,
            ) as stream:
                for token in stream.text_stream:
                    on_token(token)
                return stream.get_final_message().content[0].text.strip()
        else:
            response = self._client.messages.create(
                model=MAIN_MODEL,
                max_tokens=MAX_TOKENS_MAIN,
                system=system,
                messages=messages,
            )
            return response.content[0].text.strip()

    def _build_messages(self) -> list[dict]:
        return list(self._history)

    # 3. Fact grounding check ─────────────────────────────────────────────────

    def _fact_grounding_check(self, reply: str) -> tuple[bool, str]:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history[:-1]
        )
        facts_block = ""
        if self.state.established_facts:
            facts_block = (
                "\nEstablished facts (these are grounded — do NOT flag them):\n"
                + "\n".join(f"  • {f}" for f in self.state.established_facts)
                + "\n"
            )
        prompt = textwrap.dedent(f"""
            Conversation history so far (excluding the latest assistant reply):
            ---
            {history_text}
            ---
            {facts_block}
            Latest assistant reply:
            ---
            {reply}
            ---

            Did the assistant introduce any facts, claims, or details that are NOT
            present or directly inferable from the conversation history or
            established facts above?
            Reply with exactly one of:
              NO
              YES: <comma-separated list of introduced facts>

            Be strict. Invented names, dates, numbers, or statements count as drift.
        """).strip()

        response = self._client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS_FACTCHECK,
            messages=[{"role": "user", "content": prompt}],
        )
        detail = response.content[0].text.strip()
        flagged = detail.upper().startswith("YES")
        return flagged, detail

    # 3. Correction pass ──────────────────────────────────────────────────────

    def _correction_pass(self, original_reply: str) -> str:
        correction_instruction = textwrap.dedent(f"""
            Your previous response was flagged for potentially introducing facts
            that are not grounded in the conversation history.

            Original response:
            ---
            {original_reply}
            ---

            Please rewrite it, removing or clearly caveating any statements that
            are not directly supported by the established facts and conversation
            history. Prefix your rewrite with "{CORRECTION_TAG}".
        """).strip()

        system = self._build_system_prompt()
        messages = self._build_messages() + [
            {"role": "user", "content": correction_instruction}
        ]

        response = self._client.messages.create(
            model=MAIN_MODEL,
            max_tokens=MAX_TOKENS_MAIN,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    # 2. State update ─────────────────────────────────────────────────────────

    def _update_state(self, user_msg: str, assistant_msg: str) -> None:
        current_json = self.state.to_json()
        prompt = textwrap.dedent(f"""
            Current session state:
            {current_json}

            Latest exchange:
            USER: {user_msg}
            ASSISTANT: {assistant_msg}

            Update the state JSON with any new facts, constraints, or thread
            changes revealed in this exchange.  Return ONLY valid JSON with
            the same four keys: original_goal, established_facts,
            active_constraints, current_thread.
            Do not invent information. Only record what was explicitly stated.
        """).strip()

        response = self._client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS_HAIKU,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Extract JSON even if wrapped in markdown fences
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            updated = SessionState.from_json(json_match.group())
            # Preserve original_goal — it must not drift
            updated.original_goal = self.state.original_goal
            self.state = updated

    # 1. Context compression ──────────────────────────────────────────────────

    def _compress_history(self) -> None:
        if not self._history:
            return

        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )
        prompt = textwrap.dedent(f"""
            Summarise the following conversation into a tight factual paragraph
            that captures: decisions made, facts established, open questions, and
            the current state of the discussion.  Be concise (≤ 120 words).

            ---
            {history_text}
            ---
        """).strip()

        response = self._client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=MAX_TOKENS_HAIKU,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.content[0].text.strip()

        # Prepend to any existing compressed prefix
        if self._compressed_prefix:
            self._compressed_prefix = summary + "\n\n[Earlier]: " + self._compressed_prefix
        else:
            self._compressed_prefix = summary

        # Clear raw history — it is now encoded in the compressed prefix
        self._history = []
        print(
            f"\n  [TVASession] History compressed at turn {self._turn_count}. "
            f"Summary stored ({len(summary)} chars).\n"
        )

    # ── Public state-mutation helpers ─────────────────────────────────────────

    def add_fact(self, fact: str) -> None:
        if fact not in self.state.established_facts:
            self.state.established_facts.append(fact)

    def add_constraint(self, constraint: str) -> None:
        if constraint not in self.state.active_constraints:
            self.state.active_constraints.append(constraint)

    def remove_constraint(self, constraint: str) -> None:
        self.state.active_constraints = [
            c for c in self.state.active_constraints if c != constraint
        ]

    def reset(self, keep_goal: bool = True) -> None:
        goal = self.state.original_goal if keep_goal else ""
        self._history = []
        self._compressed_prefix = ""
        self._turn_count = 0
        self._last_was_flagged = False
        self.state = SessionState(original_goal=goal)


    def save(self, filepath: str) -> None:
        """Persist the session (state, history, compressed prefix) to a JSON file."""
        data = {
            "state": json.loads(self.state.to_json()),
            "compressed_prefix": self._compressed_prefix,
            "history": self._history,
            "turn_count": self._turn_count,
            "extra_system": self._extra_system,
        }
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, filepath: str, api_key: Optional[str] = None) -> "TVASession":
        """Restore a session from a JSON file produced by :meth:`save`."""
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
        obj = cls.__new__(cls)
        obj._client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        obj._extra_system = data.get("extra_system", "")
        obj._history = data.get("history", [])
        obj._compressed_prefix = data.get("compressed_prefix", "")
        obj._turn_count = data.get("turn_count", 0)
        obj._last_was_flagged = False
        obj.state = SessionState.from_json(json.dumps(data.get("state", {})))
        return obj

def print_meta(meta: dict) -> None:
    if meta["flagged"]:
        print(f"\n  {DRIFT_TAG}")
        print(f"  Fact-check detail: {meta['drift_detail']}")
        if meta["corrected"]:
            print("  → Correction pass ran. Response has been rewritten.")
    else:
        print(f"\n  [fact-check: clean]")


def run_cli() -> None:
    print(BANNER)

    goal = input("Enter your session goal (one sentence): ").strip()
    if not goal:
        goal = "General Q&A session"

    extra = input("Optional extra system instructions (press Enter to skip): ").strip()

    session = TVASession(goal=goal, system_prompt=extra)
    print(f"\nSession started.  Model: {MAIN_MODEL}  |  Compression threshold: {MAX_HISTORY_CHARS} chars\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "/state":
            print("\n── Current State ──")
            print(session.state.to_json())
            print("──────────────────\n")
            continue

        if user_input.lower() == "/history":
            print("\n── Raw History ──")
            for i, m in enumerate(session._history):
                role = m["role"].upper()
                content = m["content"][:200] + ("…" if len(m["content"]) > 200 else "")
                print(f"  [{i}] {role}: {content}")
            if session._compressed_prefix:
                print(f"\n  [Compressed prefix present — {len(session._compressed_prefix)} chars]")
            print("─────────────────\n")
            continue

        if user_input.lower() == "/facts":
            print("\n── Established Facts ──")
            facts = session.state.established_facts
            if facts:
                for f in facts:
                    print(f"  • {f}")
            else:
                print("  (none yet)")
            print("───────────────────────\n")
            continue

        if user_input.lower() == "/summary":
            print("\n── Compressed History Prefix ──")
            if session._compressed_prefix:
                print(session._compressed_prefix)
            else:
                print("  (no compression has run yet)")
            print("───────────────────────────────\n")
            continue

        if user_input.lower() == "/compress":
            session._compress_history()
            continue

        if user_input.lower().startswith("/save"):
            parts = user_input.split(maxsplit=1)
            filepath = parts[1] if len(parts) > 1 else "tva_session_save.json"
            session.save(filepath)
            print(f"\n  [TVASession] Session saved to '{filepath}'.\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        _, meta = session.chat(user_input, on_token=lambda t: print(t, end="", flush=True))
        print("\n")
        print_meta(meta)
        print()


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "Copy .env.example → .env and add your key, then retry."
        )
        sys.exit(1)

    run_cli()
