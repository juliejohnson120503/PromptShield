# PromptShield AI — Chrome Browser Extension (Phase 7)

Manifest V3 browser extension providing real-time, context-aware prompt interception and privacy protection directly on **ChatGPT (`chatgpt.com`)**, **Claude (`claude.ai`)**, **Google Gemini (`gemini.google.com`)**, and any LLM chat interface.

---

## Architecture

```
User Prompt (ChatGPT / Claude / Gemini / Demo)
    │
    ▼ [content.js catches submission: Enter key or Send button]
    │
    ▼ [background/service_worker.js calls FastAPI: http://localhost:8000/analyze]
    │
PromptShield Security Intercept Modal
    │  • Risk Level Badge (HIGH / MEDIUM / LOW)
    │  • Highlighted privacy preview
    │  • Entity breakdown: MASK vs RETAIN
    │
    ├── [Apply Protection & Send] ──> Calls /sanitize ──> Replaces input with tokens ──> Sends to LLM
    └── [Bypass & Send Raw]       ──> Sends raw prompt unchanged
```

---

## 3-Step Setup Guide

### 1. Start the PromptShield FastAPI Server
In your terminal, run:
```bash
python run_server.py
```
*The server will start on `http://127.0.0.1:8000` with CORS enabled for browser extensions.*

### 2. Load the Extension in Chrome / Edge / Brave
1. Open Google Chrome (or Edge / Brave / Arc).
2. Navigate to:
   ```text
   chrome://extensions
   ```
3. Toggle **"Developer mode"** in the top right corner.
4. Click the **"Load unpacked"** button in the top left.
5. Select the `extension` folder inside this repository:
   ```text
   d:\Jerin Binu\PromptShield\extension
   ```
6. The **PromptShield AI** extension will appear with a green **"ON"** badge!

### 3. Test & Demonstrate

#### Option A: Built-in Live Simulation Sandbox (Recommended for Review 2)
Open [`extension/demo_page.html`](file:///d:/Jerin%20Binu/PromptShield/extension/demo_page.html) in your browser:
* Click any pre-loaded chip (e.g. *"Personal PII vs Public Figure"* or *"Credentials & Secrets"*).
* Click **Send**.
* Watch the **PromptShield Security Intercept Modal** pop up with live risk scores and policy recommendations!
* Click **"Apply Protection & Send"** to watch the tokens get injected and processed.

#### Option B: Real ChatGPT / Claude / Gemini
1. Open [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai).
2. Type a prompt containing personal or company secrets:
   > *"My name is Julie Johnson, email julie@gmail.com. Please write an email to ABC Technologies."*
3. Press **Enter** or click **Send**.
4. PromptShield intercepts the prompt before it ever leaves your browser!

---

## Extension Structure

* [`manifest.json`](manifest.json) — Chrome Manifest V3 declaration with host permissions for localhost:8000 and LLMs.
* [`popup/`](popup/) — Interactive security control panel, level selector (Strict/Balanced/Permissive), live statistics, and quick tester.
* [`content/`](content/) — In-page prompt interceptor, glowing input badge, and full security intercept modal.
* [`background/`](background/) — Service worker handling asynchronous REST API communication with the backend.
* [`demo_page.html`](demo_page.html) — Local simulated LLM chat interface for offline/reliable live demonstration.
