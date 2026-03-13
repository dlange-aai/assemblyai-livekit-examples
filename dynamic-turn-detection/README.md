# Dynamic Turn Detection with AssemblyAI + LiveKit

This example demonstrates how to dynamically update AssemblyAI's `min_turn_silence` mid-session using the LiveKit Agents framework.

## The Problem

Relaxing end-of-utterance timing to capture full phone numbers or order IDs adds latency to every other normal utterance. A single static `min_turn_silence` value forces a tradeoff between accuracy and responsiveness.

## The Solution

Adjust `min_turn_silence` dynamically based on what kind of input the agent is expecting. `min_turn_silence` controls how long the model waits before checking if a turn is over — raising it for complex inputs like credit card numbers prevents premature end-of-turn detection during natural pauses between digit groups.

The agent uses LLM function tools to switch between modes before each prompt:

| Mode | `min_turn_silence` | Use Case |
|------|-------------------|----------|
| Yes/No | 100ms | Quick confirmations — check immediately |
| Name | 300ms | Short utterances |
| Account ID | 800ms | Alphanumeric codes |
| Phone number | 1000ms | Digit groups with pauses |
| Address | 1000ms | Multi-part with natural pauses |
| Email | 1000ms | Spelling with pauses |
| Credit card | 1500ms | 16 digits in groups of 4 |

## How It Works

1. The agent is configured with STT-based turn detection using AssemblyAI's `u3-rt-pro` model
2. Before each question, the LLM calls a function tool (e.g. `set_phone_mode`) that updates `min_turn_silence` via `assemblyai.STT.update_options()`
3. The STT waits for the configured silence duration before checking for terminal punctuation to determine if the turn is over
4. After collection is complete, the agent resets to the default (100ms)

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
- Config updates with the new `min_turn_silence` value
