# Robot voice (Inky)

The desktop turns text into speech; the Pi only plays it.

```
nav UI / desktop.voice.say ──body/voice/say──▶ desktop.voice (Kokoro-82M, RTX 5060 Ti)
                                                   │ body/audio/play (PCM chunks), body/audio/stop
                                                   ▼
                              Pi body/audio_bridge ─ pw-cat ─▶ USB speaker
                                                   └ body/audio/status (utt_started / utt_done)
```

## Setup (once)

```
uv venv --python 3.12 .venv-voice
VIRTUAL_ENV=.venv-voice uv pip install -r desktop/voice/requirements-voice.txt
```

The first run downloads the Kokoro weights from Hugging Face (~330 MB).

## Run

```
.venv-voice/bin/python -m desktop.voice --router tcp/192.168.8.60:7447
.venv/bin/python -m desktop.voice.say --router tcp/192.168.8.60:7447 "Hello"
```

The voice is set in the `voice` section of `config.json`: `robot_name` (Inky — the Phase 2
wake word), `voice` (Kokoro voice id, default
`af_heart`), `pitch` (1.0 = natural; >1 raises the pitch at the same pace) and `speed`.
`--voice/--pitch/--speed` override it for one run.

Speaker volume is PipeWire's on the Pi: `wpctl set-volume @DEFAULT_AUDIO_SINK@ 0.8`.

## What gets said

`desktop/nav/voice_client.py`: sightings (person, pet, stuffed toy, box; once per detection,
60 s per label), "I'm stuck" (BLOCKED for 4 s+), "I couldn't get there" (FAILED), "Emergency
stop", and end of an Explore run. The nav toolbar's **Voice** toggle mutes it.

Queue rules (`speech_queue.py`): alert > chat > info; an alert cuts off anything that isn't an
alert; info not started within 10 s is dropped. Each utterance's timings are logged to
`~/body-logs/voice/voice_YYYYMMDD.jsonl`.
