/**
 * PromptShield AI — Content Script (Manifest V3)
 * Real-time prompt interceptor for ChatGPT, Claude, Gemini, and LLM web interfaces.
 */

(() => {
  let isIntercepting = false;
  let bypassNextSubmit = false;
  let lastActiveInput = null;

  console.log("[PromptShield] Content script injected and active.");

  // 1. Selector utilities for LLM input boxes and submit buttons
  function findPromptInput() {
    // 1. ChatGPT
    const chatGpt = document.querySelector("#prompt-textarea, [data-id='root'] textarea");
    if (chatGpt) return chatGpt;

    // 2. Claude
    const claude = document.querySelector("div[contenteditable='true'].ProseMirror, fieldset div[contenteditable='true']");
    if (claude) return claude;

    // 3. Gemini
    const gemini = document.querySelector(".ql-editor, rich-textarea textarea, [aria-label*='prompt' i][contenteditable='true']");
    if (gemini) return gemini;

    // 4. Generic / Demo Page
    const active = document.activeElement;
    if (active && (active.tagName === "TEXTAREA" || active.getAttribute("contenteditable") === "true")) {
      return active;
    }

    return document.querySelector("textarea, div[contenteditable='true']");
  }

  function getPromptText(inputElem) {
    if (!inputElem) return "";
    if (inputElem.tagName === "TEXTAREA" || inputElem.tagName === "INPUT") {
      return inputElem.value || "";
    }
    return inputElem.innerText || inputElem.textContent || "";
  }

  function setPromptText(inputElem, text) {
    if (!inputElem) return;
    inputElem.focus();

    if (inputElem.tagName === "TEXTAREA" || inputElem.tagName === "INPUT") {
      const nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        "value"
      )?.set;
      if (nativeSetter) {
        nativeSetter.call(inputElem, text);
      } else {
        inputElem.value = text;
      }
      inputElem.dispatchEvent(new Event("input", { bubbles: true }));
      inputElem.dispatchEvent(new Event("change", { bubbles: true }));
    } else {
      // ContentEditable / ProseMirror / Lexical (ChatGPT, Claude, Gemini)
      try {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(inputElem);
        selection.removeAllRanges();
        selection.addRange(range);

        // execCommand('insertText') natively updates ProseMirror/Lexical's document state!
        const success = document.execCommand("insertText", false, text);
        if (!success) {
          inputElem.innerText = text;
        }
      } catch (err) {
        inputElem.innerText = text;
      }

      inputElem.dispatchEvent(new InputEvent("input", {
        bubbles: true,
        cancelable: true,
        inputType: "insertText",
        data: text,
      }));
      inputElem.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  function findSubmitButton() {
    // 1. ChatGPT selectors (including data-testid, aria-label, and form submit)
    const gptBtn = document.querySelector(
      "button[data-testid='send-button'], button[aria-label*='Send' i], button[data-testid*='send' i], button[data-testid='fruitjuice-send-button'], form button[type='submit']"
    );
    if (gptBtn) return gptBtn;

    // 2. Claude
    const claudeBtn = document.querySelector("button[aria-label='Send Message'], button:has(svg[viewBox*='24'])");
    if (claudeBtn) return claudeBtn;

    // 3. Gemini
    const geminiBtn = document.querySelector("button[aria-label*='Send' i], .send-button");
    if (geminiBtn) return geminiBtn;

    // 4. Generic / Sandbox
    return document.querySelector("button[type='submit'], #sendBtn, .btn-send");
  }

  // 2. Intercept Submission Events
  // Capture Enter key on input fields
  document.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      const inputElem = findPromptInput();
      if (!inputElem) return;

      // Check if event targets our input
      if (e.target === inputElem || inputElem.contains(e.target)) {
        if (bypassNextSubmit) {
          bypassNextSubmit = false;
          return;
        }

        const rawText = getPromptText(inputElem).trim();
        if (rawText.length < 5) return; // Skip trivial inputs

        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        lastActiveInput = inputElem;
        await processPromptInterception(rawText, inputElem);
      }
    }
  }, true);

  // Capture Click on Send Button
  document.addEventListener("click", async (e) => {
    const sendBtn = findSubmitButton();
    if (sendBtn && (e.target === sendBtn || sendBtn.contains(e.target))) {
      if (bypassNextSubmit) {
        bypassNextSubmit = false;
        return;
      }

      const inputElem = findPromptInput();
      if (!inputElem) return;

      const rawText = getPromptText(inputElem).trim();
      if (rawText.length < 5) return;

      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();

      lastActiveInput = inputElem;
      await processPromptInterception(rawText, inputElem);
    }
  }, true);

  // 3. Core Interception Flow
  async function processPromptInterception(promptText, inputElem) {
    if (isIntercepting) return;
    isIntercepting = true;

    showMiniStatus("PromptShield Analyzing...");

    chrome.runtime.sendMessage(
      { type: "ANALYZE_PROMPT", payload: { prompt: promptText } },
      (analyzeResponse) => {
        isIntercepting = false;
        hideMiniStatus();

        if (!analyzeResponse || !analyzeResponse.success) {
          console.warn("[PromptShield] Analysis error or offline:", analyzeResponse?.error);
          // Backend is either offline or extension is disabled; let prompt proceed
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        if (analyzeResponse.bypassed) {
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        const data = analyzeResponse.data;
        const entities = data.detected_entities || [];

        // If no sensitive entities detected, send directly without interruption
        if (entities.length === 0) {
          console.log("[PromptShield] No sensitive entities found. Passing prompt through.");
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        // Sensitive entities found! Display Security Modal
        console.log(`[PromptShield] Intercepted prompt with ${entities.length} sensitive entities.`);
        renderSecurityModal(data, promptText, inputElem);
      }
    );
  }

  function triggerDirectSubmit(text, inputElem) {
    bypassNextSubmit = true;
    const elem = (inputElem && inputElem.isConnected) ? inputElem : findPromptInput();
    if (!elem) {
      console.warn("[PromptShield] Could not locate prompt input element.");
      return;
    }

    setPromptText(elem, text);

    // Give React / ProseMirror 120ms to process the input event and enable the submit button
    setTimeout(() => {
      const sendBtn = findSubmitButton();
      if (sendBtn && !sendBtn.disabled && sendBtn.getAttribute("aria-disabled") !== "true") {
        console.log("[PromptShield] Triggering send via click on submit button.");
        sendBtn.click();
      } else {
        console.log("[PromptShield] Dispatching Enter keydown to prompt input.");
        elem.focus();
        elem.dispatchEvent(new KeyboardEvent("keydown", {
          key: "Enter",
          code: "Enter",
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
        }));
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.removeAttribute("disabled");
          sendBtn.removeAttribute("aria-disabled");
          sendBtn.click();
        }
      }
    }, 120);
  }

  // 4. Render Interactive Security Intercept Modal
  function renderSecurityModal(analysisData, originalPrompt, inputElem) {
    // Remove existing modal if any
    const existing = document.getElementById("promptshield-overlay");
    if (existing) existing.remove();

    const rawScore = analysisData.overall_risk_score || 0;
    const riskScore = rawScore <= 1.0 ? Math.round(rawScore * 100) : Math.round(rawScore);
    const policies = analysisData.entity_policies || [];

    let riskLevel = "LOW";
    const hasCriticalOrHigh = policies.some(p => {
      const r = (p.risk_level || "").toUpperCase();
      return r === "CRITICAL" || r === "HIGH" || p.entity_type === "PASSWORD" || p.entity_type === "API_KEY" || p.entity_type === "CREDIT_CARD";
    });

    if (hasCriticalOrHigh || riskScore >= 70) {
      riskLevel = "HIGH";
    } else if (riskScore >= 35 || policies.some(p => p.decision === "MASK")) {
      riskLevel = "MEDIUM";
    }

    const overlay = document.createElement("div");
    overlay.id = "promptshield-overlay";

    // Highlight sensitive words in preview
    let highlightedPrompt = escapeHtml(originalPrompt);
    policies.forEach((p) => {
      const cls = p.decision === "MASK" ? "ps-highlight-mask" : "ps-highlight-retain";
      const regex = new RegExp(`\\b${escapeRegExp(escapeHtml(p.entity_text))}\\b`, "gi");
      highlightedPrompt = highlightedPrompt.replace(regex, `<span class="${cls}">$&</span>`);
    });

    overlay.innerHTML = `
      <div id="promptshield-modal">
        <!-- Header -->
        <div class="ps-modal-header">
          <div class="ps-brand">
            <div class="ps-logo-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M12 2L3 7V12C3 17.5 7 21.5 12 22C17 21.5 21 17.5 21 12V7L12 2Z" fill="#ffffff" fill-opacity="0.2" stroke="#ffffff" stroke-width="2"/>
                <path d="M9 12L11 14L15 10" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </div>
            <div class="ps-title-group">
              <h2>PROMPTSHIELD SECURITY INTERCEPT</h2>
              <span>Context-Aware Privacy Shield • Task: ${analysisData.task_type || "GENERAL"}</span>
            </div>
          </div>
          <span class="ps-risk-pill ps-risk-${riskLevel.toLowerCase()}">${riskLevel} RISK (${Math.round(riskScore)}/100)</span>
        </div>

        <!-- Body -->
        <div class="ps-modal-body">
          <div>
            <div class="ps-section-heading">Prompt Privacy Preview</div>
            <div class="ps-prompt-preview">${highlightedPrompt}</div>
          </div>

          <div>
            <div class="ps-section-heading">Detected Sensitive Data & Policy Actions (${policies.length})</div>
            <div class="ps-entities-list">
              ${policies.map((p, idx) => `
                <div class="ps-entity-row">
                  <div class="ps-entity-meta">
                    <span class="ps-entity-name">${escapeHtml(p.entity_text)}</span>
                    <span class="ps-entity-sub">${p.entity_type} • ${escapeHtml(p.rationale || "Evaluated by policy engine")}</span>
                  </div>
                  <span class="ps-entity-decision-badge ${p.decision === "MASK" ? "ps-badge-mask" : "ps-badge-retain"}">
                    ${p.decision === "MASK" ? "🔒 MASK TOKEN" : "✓ RETAIN CONTEXT"}
                  </span>
                </div>
              `).join("")}
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="ps-modal-footer">
          <button id="psBtnBypass" class="ps-btn ps-btn-danger-ghost">Bypass & Send Raw</button>
          <div class="ps-btn-group-right">
            <button id="psBtnCancel" class="ps-btn ps-btn-ghost">Cancel & Edit</button>
            <button id="psBtnSanitize" class="ps-btn ps-btn-primary">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              Apply Protection & Send
            </button>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);

    // Event Listeners for Modal Buttons
    document.getElementById("psBtnCancel").addEventListener("click", () => {
      overlay.remove();
      inputElem.focus();
    });

    document.getElementById("psBtnBypass").addEventListener("click", () => {
      overlay.remove();
      const targetInput = (inputElem && inputElem.isConnected) ? inputElem : findPromptInput();
      triggerDirectSubmit(originalPrompt, targetInput);
    });

    document.getElementById("psBtnSanitize").addEventListener("click", () => {
      const btn = document.getElementById("psBtnSanitize");
      btn.textContent = "Sanitizing...";
      btn.disabled = true;

      chrome.runtime.sendMessage(
        { type: "SANITIZE_PROMPT", payload: { prompt: originalPrompt } },
        (sanitizeRes) => {
          overlay.remove();
          const targetInput = (inputElem && inputElem.isConnected) ? inputElem : findPromptInput();
          if (sanitizeRes && sanitizeRes.success && sanitizeRes.data) {
            const maskedText = sanitizeRes.data.sanitized_prompt;
            console.log("[PromptShield] Injected sanitized prompt into LLM:", maskedText);
            triggerDirectSubmit(maskedText, targetInput);
          } else {
            console.error("[PromptShield] Failed to sanitize prompt:", sanitizeRes);
            triggerDirectSubmit(originalPrompt, targetInput);
          }
        }
      );
    });
  }

  // 5. Floating Status Pill Helper
  function showMiniStatus(text) {
    let pill = document.getElementById("promptshield-mini-status");
    if (!pill) {
      pill = document.createElement("div");
      pill.id = "promptshield-mini-status";
      pill.style.cssText = `
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        z-index: 999999 !important;
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid #38bdf8 !important;
        color: #38bdf8 !important;
        padding: 6px 14px !important;
        border-radius: 999px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        font-family: sans-serif !important;
      `;
      document.body.appendChild(pill);
    }
    pill.innerHTML = `
      <span style="width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; display: inline-block; animation: pulse 1s infinite;"></span>
      ${text}
    `;
    pill.style.display = "flex";
  }

  function hideMiniStatus() {
    const pill = document.getElementById("promptshield-mini-status");
    if (pill) pill.style.display = "none";
  }

  function escapeHtml(str) {
    return (str || "").replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  }

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
})();
