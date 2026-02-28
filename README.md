# drift-lock

**drift-lock** is an open-source Python library and web app that wraps the Anthropic API to actively prevent LLM hallucination. It intercepts every model response, audits it for unsupported facts, corrects drift automatically, and keeps a structured session state that anchors the model to what has actually been established in the conversation.

---

## The Problem

Large language models hallucinate. Even in a focused multi-turn conversation, a model will gradually introduce unverified claims, misremember earlier facts, and drift away from the original goal. Standard chat wrappers do nothing to detect or correct this.

## How drift-lock Solves It

Every response passes through four mechanisms before it reaches the user:

### 1. Fact Grounding Check
After each reply, a fast secondary model call audits the response:
- **Did the assistant introduce any facts not present in the conversation history or established state?**
- If **YES** → the response is flagged and a correction pass is triggered automatically.
- If **NO** → the response is returned as-is with a `✓ Grounded` badge.

### 2. Automatic Correction Pass
When drift is detected, the model is re-prompted with the original response and asked to rewrite it — removing or caveating any unsupported claims. The corrected version replaces the original in both the UI and the session history.

### 3. Structured Session State
After every turn, the session state is updated by extracting:
- **established_facts** — facts explicitly stated in the conversation
- **active_constraints** — rules or constraints the user has set
- **current_thread** — what the conversation is currently about
- **original_goal** — the user's stated goal (immutable, never drifts)

This state is injected into every system prompt, so the model always knows what has been grounded.

### 4. Anchor Injection
The original goal and all established facts are prepended to every system prompt automatically — zero extra API calls. This keeps long conversations anchored to the original intent regardless of how many turns have passed.

### 5. Context Compression (need-based)
When raw conversation history exceeds a character threshold, it is summarised into a tight factual paragraph and the raw turns are cleared. Compression is triggered by actual context size — not a fixed turn counter — so short exchanges are never compressed unnecessarily.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser                            │
│              React + Vite frontend                      │
│   SetupScreen → ChatScreen → MessageBubble + StatePanel │
└────────────────────┬────────────────────────────────────┘
                     │  /api/*  (SSE streaming)
┌────────────────────▼────────────────────────────────────┐
│                   Flask backend                         │
│                     app.py                              │
│         /api/start    /api/chat  (SSE)                  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  TVASession                             │
│               tva_session.py                            │
│                                                         │
│  chat() ──► _call_main()          (main model)          │
│         ──► _fact_grounding_check (haiku audit)         │
│         ──► _correction_pass()    (if drift found)      │
│         ──► _update_state()       (haiku extraction)    │
│         ──► _compress_history()   (if context > limit)  │
└────────────────────┬────────────────────────────────────┘
                     │
              Anthropic API
           claude-haiku-4-5
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- An [Anthropic API key](https://console.anthropic.com/settings/keys)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/drift-lock.git
cd drift-lock
```

### 2. Set up the Python backend

```bash
pip install -r requirements.txt
```

Copy the example env file and add your API key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Set up the React frontend

```bash
cd frontend
npm install
```

### 4. Run in development

Open two terminals:

```bash
# Terminal 1 — Flask backend
PORT=5001 python app.py

# Terminal 2 — Vite dev server
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

The Vite dev server proxies all `/api/*` requests to Flask at port 5001 automatically.

---

## Production Build

```bash
cd frontend && npm run build
cd ..
python app.py   # serves both the React build and the API on port 5000
```

---

## Project Structure

```
drift-lock/
├── tva_session.py        # Core TVASession class — use this standalone
├── app.py                # Flask web server (SSE streaming)
├── requirements.txt      # Python dependencies
├── Procfile              # Railway / Render deployment
├── .env.example          # Environment variable template
└── frontend/
    ├── src/
    │   ├── App.jsx                    # Session state + streaming logic
    │   ├── index.css                  # Tailwind + custom animations
    │   └── components/
    │       ├── SetupScreen.jsx        # API key + goal form
    │       ├── ChatScreen.jsx         # Chat layout + input bar
    │       ├── MessageBubble.jsx      # Message + fact-check badge
    │       └── StatePanel.jsx         # Live session state panel
    ├── package.json
    └── vite.config.js                 # Dev proxy to Flask
```

---

## Using TVASession Directly

`TVASession` is a standalone Python class. Use it without the web app:

```python
from tva_session import TVASession

session = TVASession(
    goal="Understand the causes of World War 1",
    system_prompt="Keep answers concise.",  # optional
)

# Basic chat
reply, meta = session.chat("What sparked the war?")
print(reply)
print(meta)
# {
#   "flagged": False,
#   "corrected": False,
#   "drift_detail": "NO",
#   "turn_count": 1
# }

# Streaming
reply, meta = session.chat(
    "Who were the main alliances?",
    on_token=lambda t: print(t, end="", flush=True),
)

# Inject grounded facts manually
session.add_fact("The assassination of Archduke Franz Ferdinand occurred in 1914.")
session.add_constraint("Only reference events before 1920.")

# Save and restore
session.save("my_session.json")
loaded = TVASession.load("my_session.json")

# Reset while keeping the goal
session.reset(keep_goal=True)
```

### `chat()` return value

| Key | Type | Description |
|-----|------|-------------|
| `flagged` | `bool` | `True` if the response contained unsupported facts |
| `corrected` | `bool` | `True` if a correction pass rewrote the response |
| `drift_detail` | `str` | The fact-check model's raw output (`NO` or `YES: ...`) |
| `turn_count` | `int` | Current turn number (1-based) |

### Public methods

| Method | Description |
|--------|-------------|
| `chat(msg, on_token=None)` | Send a message, get `(reply, meta)` |
| `add_fact(fact)` | Inject a grounded fact into session state |
| `add_constraint(constraint)` | Add a constraint |
| `remove_constraint(constraint)` | Remove a constraint by exact match |
| `reset(keep_goal=True)` | Clear history and state |
| `save(filepath)` | Persist session to JSON |
| `load(filepath)` | Restore session from JSON (classmethod) |

---

## Configuration

| Constant | Default | Description |
|----------|---------|-------------|
| `MAIN_MODEL` | `claude-haiku-4-5` | Model used for main responses |
| `HAIKU_MODEL` | `claude-haiku-4-5` | Model used for fact-check and state update |
| `MAX_HISTORY_CHARS` | `6000` | Raw history size before compression triggers |
| `MAX_TOKENS_MAIN` | `1024` | Max tokens for main model responses |
| `MAX_TOKENS_HAIKU` | `512` | Max tokens for state update calls |
| `MAX_TOKENS_FACTCHECK` | `256` | Max tokens for fact-check calls |

---

## CLI

You can also run drift-lock as a terminal chat:

```bash
python tva_session.py
```

Available commands during a session:

| Command | Description |
|---------|-------------|
| `/state` | Show full session state JSON |
| `/history` | Show raw message history |
| `/facts` | List established facts |
| `/summary` | Show the compressed history prefix |
| `/compress` | Manually trigger history compression |
| `/save [file]` | Save session to a JSON file |
| `/quit` | Exit |

---

## Deployment

### Frontend → Vercel

1. Push the repo to GitHub
2. Go to [vercel.com](https://vercel.com) → **New Project** → import the repo
3. Set the following in Vercel project settings:
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add environment variable:
   - `VITE_API_BASE_URL` → your Railway backend URL (see below)
5. Deploy

### Backend → Railway

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub**
2. Select the repo
3. Add environment variable:
   - `ANTHROPIC_API_KEY` → your key *(users provide their own key via the UI — this is only needed if you want a fallback)*
4. Railway auto-detects the `Procfile` and deploys Flask

Once Railway gives you a URL (e.g. `https://drift-lock.up.railway.app`), paste it into the `VITE_API_BASE_URL` env var on Vercel and redeploy.

---

## How the Fact-Grounding Check Works

The auditing prompt sent to the secondary model:

```
Conversation history so far (excluding the latest assistant reply):
---
[full history]
---
Established facts (these are grounded — do NOT flag them):
  • fact 1
  • fact 2
---
Latest assistant reply:
---
[reply text]
---

Did the assistant introduce any facts, claims, or details that are NOT
present or directly inferable from the conversation history or
established facts above?

Reply with exactly one of:
  NO
  YES: <comma-separated list of introduced facts>
```

If the response starts with `YES`, the correction pass fires.

---

## Contributing

Pull requests are welcome. To contribute:

1. Fork the repo
2. Create a feature branch: `git checkout -b my-feature`
3. Commit your changes
4. Open a pull request

---

## License

MIT
