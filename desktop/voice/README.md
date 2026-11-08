# Robot voice (Inky)

The desktop turns text into speech and listens; the Pi only plays and records.

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

During **Explore** (`on_explore_*` in voice_client.py): "Okay, I'm going exploring."; a
new thing the VLM labelled `other`/`furniture` with confidence ≥ 0.6 → "I found something
new: a blue laundry basket." (its few-word `what`, at most one per 20 s); before an
investigate leg "I'm going to take a closer look at that box."; the result ("Yes, there
really is a box there." / "it's gone now" / "False alarm" / "couldn't get close enough");
"Heading home now."; and a summary: "Exploration finished. I looked over 72 percent of the
house and found 2 boxes and a stuffed toy." Places the change store knows recur are never
announced.

`desktop/nav/voice_client.py`: sightings (person, pet, stuffed toy, box; once per detection,
60 s per label), "I'm stuck" (BLOCKED for 4 s+), "I couldn't get there" (FAILED), "Emergency
stop", and end of an Explore run. The nav toolbar's **Voice** toggle mutes it.

Queue rules (`speech_queue.py`): alert > chat > info; an alert cuts off anything that isn't an
alert; info not started within 10 s is dropped. Each utterance's timings are logged to
`~/body-logs/voice/voice_YYYYMMDD.jsonl`.

## Chat ("Inky, ...")

```
Pi mic ─pw-record 16 kHz─▶ body/audio/mic (muted while the speaker plays + 0.4 s)
   ▼ desktop.voice listener
Silero VAD (CPU) → segment (0.7 s of silence ends it) → Whisper large-v3-turbo (5060 Ti)
   → wake.py: "Inky" near the start, fuzzy ("Inkie", "Enki", "in key", "Hey Inky")
   → chat.py: LLM via desktop/vision_service.chat (the vLLM server) → spoken as kind=chat
```

- Name alone ("Inky?") → "Yes?". After Inky finishes a reply or a "Yes?", anything said
  within `follow_up_s` (8 s) is answered without the name. History: last 6 exchanges,
  cleared after 3 min quiet.
- Half-duplex: the mic is muted while Inky talks, so you can't interrupt it yet.
- Every transcribed segment (text only, no audio) goes to
  `~/body-logs/voice/heard_YYYYMMDD.jsonl` and `body/voice/heard`.
- Settings: `voice.listen` in `config.json` (persona, follow-up window, segment timing);
  `audio.mic_*` / `mute_tail_s` for the Pi side. `--no-listen` runs speech only.
- Setup addition: `VIRTUAL_ENV=.venv-voice uv pip install silero-vad`.
