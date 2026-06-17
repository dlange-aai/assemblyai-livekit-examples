# Including `agent_context` — minimal AssemblyAI + LiveKit agent

A bare-bones voice assistant showing the smallest amount of code needed to feed
AssemblyAI's `agent_context` from a LiveKit agent. Added in **livekit-agents
1.6.0**, `agent_context` tells the `u3-rt-pro` streaming model what the agent just
said so it can bias transcription of the user's *next* reply.

## The handler

```python
@session.on("conversation_item_added")
def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
    if ev.item.type != "message" or ev.item.role != "assistant":
        return
    agent_stt = session.stt
    if not isinstance(agent_stt, assemblyai.STT):
        return
    spoken = ev.item.text_content
    if spoken:
        agent_stt.update_options(agent_context=spoken[-1500:])
```

- Fires whenever a message is added to the conversation; filter to the agent's
  own turns (`ev.item.role == "assistant"`).
- `update_options(agent_context=...)` sends an `UpdateConfiguration` over the live
  STT websocket — no reconnect.
- Only supported on the `u3-rt-pro` family. Max **1500 characters**.
- Push just the latest utterance; the plugin carries earlier turns forward
  automatically via `previous_context_n_turns` (a construction-time option on
  `assemblyai.STT`; leave unset for the server default, or `0` to disable).

## Setup

```bash
pip install "livekit-agents[assemblyai,silero]~=1.6" python-dotenv
```

Copy the repo-root `.env.example` to `.env` and fill in your keys:

```
ASSEMBLYAI_API_KEY=your_key
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
```

LLM and TTS run through LiveKit Inference, so no separate provider keys are needed.

## Run

```bash
python agent.py dev
```

Then connect via the [LiveKit Agents Playground](https://agents-playground.livekit.io).
