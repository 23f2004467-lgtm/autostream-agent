# Demo recording — AutoStream Agent

Target length: **2:00 – 2:45**. Short enough that reviewers actually
watch it, long enough to hit every rubric beat.

## Pre-flight (do once before recording)

- [ ] **Restart the Streamlit Cloud app** (Manage app → Reboot) so
      you start from a fresh session with the sidebar inspector empty.
- [ ] Open **two tabs** arranged side-by-side:
  1. Landing page — <https://23f2004467-lgtm.github.io/autostream-agent/design/landing/Landing%20Page.html>
  2. Streamlit app — <https://autostream-agent-yor2zpfuuypeitjohgtmml.streamlit.app/>
- [ ] Open **Streamlit Cloud logs** in a third tab or split so the
      "Lead captured successfully:" terminal line is on screen when
      the tool fires. (Manage app → Logs pane.)
- [ ] Open VS Code with the repo. Pre-navigate to these files in
      separate tabs so you can flash through them at 2:00:
  - `src/graph.py` (scroll to `after_classify` ~line 240)
  - `src/tools.py` (whole file — 25 lines)
  - `tests/test_flow.py` (show the list of test names)
  - `README.md` (scroll to the WhatsApp deployment section)
- [ ] Bump font size in VS Code (Cmd-+ 2-3x) so code is readable at
      1080p.
- [ ] Record with QuickTime (⌘⇧5 → Record Selected Portion) or OBS.
      Full desktop, not a specific window, so tab-switching is
      visible. Include system audio OFF, mic ON.

## Recording script (read while demoing)

### 0:00 — Landing page (8 sec)

Show landing tab.

> "AutoStream — a conversational sales agent I built for ServiceHive's
> ML intern assignment. This is the landing page; the agent is one tap
> away."

Click **"Chat with the agent →"**. Cut to Streamlit.

### 0:08 — Greeting (10 sec)

Copy-paste: `hi`

> "Greeting intent. The inspector on the left shows everything the
> agent knows — phase, last classified intent, the three lead slots
> it's trying to fill, and any captured leads this session."

### 0:18 — RAG (22 sec)

Copy-paste: `tell me about pricing`

> "Product-inquiry intent. RAG kicks in — FAISS index over a
> chunked markdown knowledge base. Notice the agent answers with
> exact plan prices and features pulled from the KB — no
> hallucination."

Wait for the answer. Point at the inspector — `LAST INTENT:
product_inquiry`.

### 0:40 — Intent shift (15 sec)

Copy-paste: `I want to sign up for Pro for my YouTube channel`

> "High-intent plus a platform mention in one message. Phase
> transitions from browsing to qualifying. The extractor already
> caught `YouTube` and filled the platform slot — see the inspector."

Pause. `PLATFORM` row should be orange with "YouTube".

### 0:55 — Lead qualification (30 sec)

Agent asks for name. Copy-paste: `Jainam`

> "Name slot fills."

Agent asks for email. Copy-paste: `jainam@distill.fyi`

> "Email's regex-validated before the slot counts. All three slots
> are now green."

Agent should now show the confirmation prompt with two chips:
**"Yes, submit"** and **"Fix something"**.

### 1:25 — Tool fire (15 sec)

Click **Yes, submit** chip.

Immediately point at the terminal / logs tab:

> "`Lead captured successfully: Jainam, jainam@distill.fyi, YouTube`
> — exact signature and print string from the brief. The inspector's
> captures feed now shows 1 LEAD. Tool fired exactly once, only after
> explicit consent."

### 1:40 — Multi-turn memory proof (15 sec)

Copy-paste: `what was my name again?`

> "LangGraph's MemorySaver checkpointer keeps the full history under
> a per-session thread ID — zero manual buffer code. The agent pulls
> the name from turn 3."

### 1:55 — Code flash (25 sec)

Cut to VS Code. Flash through tabs, 5 seconds each, narrating:

1. `src/graph.py` — `after_classify` / phase routing:
   > "Phase-driven LangGraph with five nodes: classify, extract,
   > retrieve, capture, respond. Phase — not intent — drives
   > routing, so a bare `Jainam` during qualifying isn't misread
   > as a greeting."

2. `src/tools.py` — 25-line file:
   > "`mock_lead_capture` — exact signature and print string from
   > the brief."

3. `tests/test_flow.py` — show the 14 test names in the outline:
   > "14 pytest cases covering intent, extraction, corrections,
   > hallucination guard, tool-gating, and six-turn memory."

### 2:20 — README / deployability (20 sec)

Cut to README, scroll to **WhatsApp Deployment** section.

> "For production: identical graph behind a webhook, `thread_id =
> WhatsApp phone number`, and the `quick_replies` state field maps
> one-to-one onto Meta's interactive-button contract. Zero changes
> to the agent code itself."

### 2:40 — Close (5 sec)

> "Thanks for watching. Repo and live URLs in the README."

## Exact user inputs (for copy-paste)

```
hi
tell me about pricing
I want to sign up for Pro for my YouTube channel
Jainam
jainam@distill.fyi
what was my name again?
```

After "jainam@distill.fyi" the agent will confirm. Click the
**"Yes, submit"** chip (don't type) to show the button pathway.

## If something breaks mid-take

- Agent mis-classifies or loops → reboot the app, start over.
- Tool fires early → would be a regression; tell me, don't edit
  mid-demo.
- Quota 429 from Groq → wait 60 s, restart the browser tab.
- Streamlit Cloud cold-started ("Loading…" forever) → the free tier
  sleeps; just click "Wake up" and retry after ~45 s.

## Upload + submission

1. Export the recording as MP4, 1080p, <60 MB if possible.
2. Upload to YouTube unlisted OR Loom OR Google Drive with link
   sharing enabled.
3. Email ServiceHive with the repo URL and video link:
   - Repo: <https://github.com/23f2004467-lgtm/autostream-agent>
   - Live app: <https://autostream-agent-yor2zpfuuypeitjohgtmml.streamlit.app/>
   - Demo video: <paste link>
