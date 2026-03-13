"""
Dynamic Turn Detection with AssemblyAI + LiveKit
=================================================

This example demonstrates how to dynamically update AssemblyAI's
max_turn_silence mid-session using the LiveKit Agents framework.

The problem: relaxing end-of-utterance timing to capture full phone
numbers or order IDs adds latency to every other normal utterance.
This example shows how to solve that by adjusting max_turn_silence
based on what kind of input the agent is expecting.

Use case: A contact center voice agent that collects a caller's name,
phone number, and PIN — each step uses a different max_turn_silence
for the right balance of accuracy vs. latency.

Requirements:
    pip install "livekit-agents[assemblyai,anthropic,cartesia,silero]~=1.4"

Environment variables:
    ASSEMBLYAI_API_KEY
    CARTESIA_API_KEY
    LIVEKIT_API_KEY
    LIVEKIT_API_SECRET
    LIVEKIT_URL
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.voice import ModelSettings
from livekit.plugins import assemblyai, anthropic, cartesia, silero

load_dotenv()
logger = logging.getLogger("dynamic-config-example")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Transcript logger — writes to both console and a JSONL file
# ---------------------------------------------------------------------------
LOG_FILE = Path("transcription_log.jsonl")


def log_event(event_type: str, **data):
    """Log an event to console and append to JSONL file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    logger.info("[%s] %s", event_type, json.dumps(data, default=str))
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_stt_event(ev: SpeechEvent):
    """Log a raw STT event with all available data."""
    if ev.type == SpeechEventType.START_OF_SPEECH:
        log_event("START_OF_SPEECH")

    elif ev.type == SpeechEventType.END_OF_SPEECH:
        log_event("END_OF_SPEECH")

    elif ev.type == SpeechEventType.RECOGNITION_USAGE:
        log_event(
            "RECOGNITION_USAGE",
            audio_duration=ev.recognition_usage.audio_duration if ev.recognition_usage else None,
        )

    elif ev.type in (
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.PREFLIGHT_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
    ):
        alt = ev.alternatives[0] if ev.alternatives else None
        if not alt:
            return

        words = None
        if alt.words:
            words = [
                {
                    "word": str(w),
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": w.confidence,
                }
                for w in alt.words
            ]

        log_event(
            ev.type.value,
            text=alt.text,
            confidence=alt.confidence,
            language=str(alt.language) if alt.language else None,
            start_time=alt.start_time,
            end_time=alt.end_time,
            speaker_id=alt.speaker_id,
            words=words,
        )


class FormFillingAgent(Agent):
    """
    A voice agent that collects caller info in steps, dynamically
    updating max_turn_silence at each step so longer inputs (like
    phone numbers) don't get cut off, while keeping latency low
    for short responses.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful contact center agent. "
                "Your job is to collect the following information from the caller, in this order:\n"
                "1. Ask if they are an existing customer (yes/no)\n"
                "2. Full name\n"
                "3. Phone number (10 digits)\n"
                "4. Confirm the phone number back (yes/no)\n"
                "5. Street address (full mailing address)\n"
                "6. Confirm the address back (yes/no)\n"
                "7. Email address\n"
                "8. Confirm the email back (yes/no)\n"
                "9. Account ID (alphanumeric code)\n"
                "10. Credit card number (16 digits)\n"
                "11. Confirm the credit card number back (yes/no)\n"
                "Before each step, call the appropriate tool to configure turn detection. "
                "Confirm each piece of information before moving to the next step. "
                "Be patient and friendly."
            ),
        )

    # ------------------------------------------------------------------
    # Override stt_node to intercept and log all raw STT events
    # ------------------------------------------------------------------

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            log_stt_event(ev)
            yield ev

    # ------------------------------------------------------------------
    # Tools that the LLM can call — each one updates max_turn_silence
    # before the next user utterance.
    # ------------------------------------------------------------------

    @agents.function_tool()
    async def set_yes_no_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking a yes/no or short confirmation question."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=500,  # Single word — respond as fast as possible
        )
        log_event("config_update", step="yes_no", max_turn_silence=500)
        return "Turn detection set for quick yes/no response."

    @agents.function_tool()
    async def set_name_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for the caller's name."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=1000,  # Names are short utterances
        )
        log_event("config_update", step="name", max_turn_silence=1000)
        return "Turn detection set for name input."

    @agents.function_tool()
    async def set_phone_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for a phone number."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=4000,  # Callers pause between digit groups
        )
        log_event("config_update", step="phone", max_turn_silence=4000)
        return "Turn detection set for phone number input. Will wait for pauses between digit groups."

    @agents.function_tool()
    async def set_address_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for a street/mailing address."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=4000,  # Addresses are multi-part with natural pauses
        )
        log_event("config_update", step="address", max_turn_silence=4000)
        return "Turn detection set for address input. Will wait for pauses between address parts."

    @agents.function_tool()
    async def set_email_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for an email address."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=4000,  # People spell emails slowly with pauses
        )
        log_event("config_update", step="email", max_turn_silence=4000)
        return "Turn detection set for email input. Will wait for spelling pauses."

    @agents.function_tool()
    async def set_account_id_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for an account ID or alphanumeric code."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=3000,  # Alphanumeric codes — people read off screens
        )
        log_event("config_update", step="account_id", max_turn_silence=3000)
        return "Turn detection set for account ID input."

    @agents.function_tool()
    async def set_credit_card_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before asking for a credit card number."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=5000,  # 16 digits in groups of 4 — long pauses between groups
        )
        log_event("config_update", step="credit_card", max_turn_silence=5000)
        return "Turn detection set for credit card input. Will wait for pauses between digit groups."

    @agents.function_tool()
    async def finish_collection(self, context: agents.RunContext[None]) -> str:
        """Call this after all info has been collected and confirmed."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            max_turn_silence=1000,  # Back to default for general conversation
        )
        log_event("config_update", step="done", max_turn_silence=1000)
        return "Collection complete. You can now help the caller with their request."


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint — sets up the session with AssemblyAI U3 Pro."""
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(
            model="u3-rt-pro",
            min_turn_silence=100,
            max_turn_silence=1000,
            vad_threshold=0.3,
        ),
        tts=cartesia.TTS(model="sonic-3"),
        llm=anthropic.LLM(model="claude-sonnet-4-20250514"),
        vad=silero.VAD.load(activation_threshold=0.3),
        turn_detection="stt",
        min_endpointing_delay=0,
    )

    await session.start(
        room=ctx.room,
        agent=FormFillingAgent(),
    )

    log_event("session_started", room=ctx.room.name)

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
