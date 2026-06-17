import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import assemblyai, silero

logger = logging.getLogger("including-agent-context")

load_dotenv()

# Keep AssemblyAI's server-side VAD and the local Silero VAD at the same
# sensitivity — a mismatch creates a dead zone that delays interruptions.
VAD_ACTIVATION_THRESHOLD = 0.3

# `agent_context` accepts up to 1500 characters per the AssemblyAI streaming API.
AGENT_CONTEXT_MAX_CHARS = 1500


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant. Keep your responses concise "
                "and conversational. Do not use emojis, asterisks, or markdown."
            )
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and ask how you can help."
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session: AgentSession = AgentSession(
        # AssemblyAI's Universal-3.5 Pro streaming model (served by the
        # `u3-rt-pro-beta-1` id below). `agent_context` is only supported on this
        # family, so it must be the STT here.
        stt=assemblyai.STT(
            model="u3-rt-pro-beta-1",
            prompt="Transcribe this customer service call.",
            min_turn_silence=100,
            max_turn_silence=400,
            vad_threshold=VAD_ACTIVATION_THRESHOLD,
        ),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        vad=silero.VAD.load(activation_threshold=VAD_ACTIVATION_THRESHOLD),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            endpointing={"min_delay": 0},
        ),
    )

    # After each agent reply, push what the assistant just said into AssemblyAI's
    # `agent_context`. The model uses it to bias transcription of the user's *next*
    # reply toward what makes sense as an answer to that prompt (e.g. after "what's
    # your email?", it leans toward an email address). The update is sent live over
    # the STT websocket via update_options() — no reconnect.
    #
    # There is no enable_agent_context() helper; this handler is the pattern.
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        # Only the agent's own spoken turns become context for the user's reply.
        if ev.item.type != "message" or ev.item.role != "assistant":
            return

        agent_stt = session.stt
        if not isinstance(agent_stt, assemblyai.STT):
            return

        spoken = ev.item.text_content
        if not spoken:
            return

        # Push only the latest agent utterance; the plugin carries earlier turns
        # forward automatically (see previous_context_n_turns on assemblyai.STT).
        agent_stt.update_options(agent_context=spoken[-AGENT_CONTEXT_MAX_CHARS:])
        logger.info(f"updated agent_context for next turn ({len(spoken)} chars)")

    await session.start(agent=Assistant(), room=ctx.room, room_options=room_io.RoomOptions())


if __name__ == "__main__":
    cli.run_app(server)
