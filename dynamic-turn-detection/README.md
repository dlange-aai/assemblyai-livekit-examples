# Dynamic Turn Detection with AssemblyAI + LiveKit

This example demonstrates how to dynamically update AssemblyAI's `max_turn_silence` mid-session using the LiveKit Agents framework.

## The Problem

Relaxing end-of-utterance timing to capture full phone numbers or order IDs adds latency to every other normal utterance. A single static `max_turn_silence` value forces a tradeoff between accuracy and responsiveness.

## The Solution

Adjust `max_turn_silence` dynamically based on what kind of input the agent is expecting. The agent uses LLM function tools to switch between modes before each prompt:

| Mode | `max_turn_silence` | Use Case |
|------|-------------------|----------|
| Yes/No | 500ms | Quick confirmations |
| Name | 1000ms | Short utterances |
| Account ID | 3000ms | Alphanumeric codes |
| Phone number | 4000ms | Digit groups with pauses |
| Address | 4000ms | Multi-part with natural pauses |
| Email | 4000ms | Spelling with pauses |
| Credit card | 5000ms | 16 digits in groups of 4 |

## How It Works

1. The agent is configured with STT-based turn detection using AssemblyAI's `u3-rt-pro` model
2. Before each question, the LLM calls a function tool (e.g. `set_phone_mode`) that updates `max_turn_silence` via `assemblyai.STT.update_options()`
3. The STT applies the new silence threshold immediately for the next utterance
4. After collection is complete, the agent resets to default timing

## Setup

### Requirements

```bash
pip install "livekit-agents[assemblyai,anthropic,cartesia,silero]~=1.4"
```

### Environment Variables

Create a `.env` file in the project root:

```
ASSEMBLYAI_API_KEY=your_key
CARTESIA_API_KEY=your_key
LIVEKIT_API_KEY=your_key
LIVEKIT_API_SECRET=your_secret
LIVEKIT_URL=wss://your-project.livekit.cloud
ANTHROPIC_API_KEY=your_key
```

### Run

```bash
python dynamic_config_livekit_example.py dev
```

Then connect via the [LiveKit Agents Playground](https://agents-playground.livekit.io).

## Logging

All raw STT events are logged to both the console and `transcription_log.jsonl`, including:

- `START_OF_SPEECH` / `END_OF_SPEECH`
- `INTERIM_TRANSCRIPT` / `PREFLIGHT_TRANSCRIPT` / `FINAL_TRANSCRIPT`
- `RECOGNITION_USAGE`
- Config updates with the new `max_turn_silence` value
