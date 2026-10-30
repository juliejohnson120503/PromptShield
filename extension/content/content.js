/**
 * PromptShield AI — Content Script (Manifest V3)
 * Real-time prompt interceptor for ChatGPT, Claude, Gemini, and LLM web interfaces.
 */

(() => {
  let isIntercepting = false;
  let bypassUntil = 0; // Timestamp window during which PromptShield does not re-intercept
  let lastActiveInput = null;
  let lastSessionId = null;

  console.log("[PromptShield] Content script injected and active.");

  // Helper: verify Chrome extension runtime context is still connected
  function isExtensionContextValid() {
    return typeof chrome !== "undefined" && Boolean(chrome?.runtime?.id) && typeof chrome.runtime?.sendMessage === "function";
  }

  // Safe wrapper around chrome.runtime.sendMessage that gracefully handles extension reloads & context invalidation
  function safeSendMessage(message, callback) {
    if (!isExtensionContextValid()) {
      console.warn("[PromptShield] Extension runtime disconnected or context invalidated. Please refresh the page (F5).");
      if (callback) callback({ success: false, error: "EXTENSION_CONTEXT_INVALIDATED" });
      return;
    }
    try {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime?.lastError) {
          console.warn("[PromptShield] Runtime message error:", chrome.runtime.lastError.message);
          if (callback) callback({ success: false, error: chrome.runtime.lastError.message });
          return;
        }
        if (callback) callback(response);
      });
    } catch (err) {
      console.warn("[PromptShield] sendMessage threw:", err);
      if (callback) callback({ success: false, error: err?.message || "EXTENSION_SEND_ERROR" });
    }
  }

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

    // 1. If input already contains the EXACT desired text, DO NOT touch it!
    const current = getPromptText(inputElem).trim();
    if (current === text.trim()) {
      return;
    }

    inputElem.focus();

    if (inputElem.tagName === "TEXTAREA" || inputElem.tagName === "INPUT") {
      const proto = window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const nativeSetter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
      if (nativeSetter) {
        nativeSetter.call(inputElem, text);
      } else {
        inputElem.value = text;
      }
      inputElem.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
      inputElem.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    } else {
      // ContentEditable / ProseMirror / Lexical (ChatGPT, Claude, Gemini)
      inputElem.focus();

      // Step 1: Select all existing content and replace using execCommand
      let replaced = false;
      try {
        document.execCommand("selectAll", false, null);
        replaced = document.execCommand("insertText", false, text);
      } catch (_) {}

      // Step 2: If execCommand didn't replace cleanly, clear ProseMirror nodes and set text
      if (!replaced || getPromptText(inputElem).trim() !== text.trim()) {
        try {
          inputElem.innerHTML = "<p><br></p>";
          const p = inputElem.querySelector("p");
          if (p) {
            p.textContent = text;
          }
        } catch (_) {}
      }

      // Step 3: If still not exact, use simulated paste on the cleared editor
      if (getPromptText(inputElem).trim() !== text.trim()) {
        try {
          document.execCommand("selectAll", false, null);
          document.execCommand("delete", false, null);
          const dt = new DataTransfer();
          dt.setData("text/plain", text);
          const pasteEvt = new ClipboardEvent("paste", {
            clipboardData: dt,
            bubbles: true,
            cancelable: true,
            composed: true,
          });
          inputElem.dispatchEvent(pasteEvt);
        } catch (_) {}
      }

      // Step 4: Final DOM cleanup fallback
      if (getPromptText(inputElem).trim() !== text.trim()) {
        inputElem.textContent = text;
      }

      // Step 5: Dispatch synthetic input and change events with composed: true
      inputElem.dispatchEvent(new InputEvent("input", {
        bubbles: true,
        cancelable: true,
        composed: true,
        inputType: "insertText",
        data: text,
      }));
      inputElem.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
      inputElem.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    }
  }

  function findSubmitButton(inputElem) {
    // 1. Search within the active form, fieldset, or composer container first
    if (inputElem) {
      const container = inputElem.closest("form, fieldset, [class*='composer'], main");
      if (container) {
        const scopedBtn = container.querySelector(
          "button[data-testid='send-button'], button[data-testid='composer-send-button'], button[aria-label*='Send prompt' i], button[aria-label*='Send message' i], button[aria-label*='Send' i], button[type='submit']"
        );
        if (scopedBtn) return scopedBtn;
      }
    }

    // 2. Global ChatGPT selectors (including testid, aria-labels, and buttons)
    const gptBtn = document.querySelector(
      "button[data-testid='send-button'], button[data-testid='composer-send-button'], button[data-testid*='send' i], button[aria-label*='Send prompt' i], button[aria-label*='Send message' i], button[aria-label*='Send' i], button#send-button, button[data-testid='fruitjuice-send-button'], form button[type='submit'], fieldset button[aria-label*='Send' i]"
    );
    if (gptBtn) return gptBtn;

    // 3. Claude
    const claudeBtn = document.querySelector("button[aria-label='Send Message'], button[aria-label*='Send' i], button:has(svg[viewBox*='24'])");
    if (claudeBtn) return claudeBtn;

    // 4. Gemini
    const geminiBtn = document.querySelector("button[aria-label*='Send' i], .send-button");
    if (geminiBtn) return geminiBtn;

    // 5. Generic / Sandbox
    return document.querySelector("button[type='submit'], #sendBtn, .btn-send");
  }

  function isSendButtonClicked(target) {
    if (!target) return false;

    // CRITICAL: PromptShield's own modal and overlay buttons must NEVER be treated as LLM send buttons!
    if (target.closest("#promptshield-modal, #promptshield-overlay, #promptshield-floating-indicator, .ps-btn")) {
      return false;
    }

    const btn = target.closest("button");
    if (!btn) return false;

    const testId = (btn.getAttribute("data-testid") || "").toLowerCase();
    const ariaLabel = (btn.getAttribute("aria-label") || "").toLowerCase();
    const id = (btn.id || "").toLowerCase();

    if (
      testId.includes("send") ||
      ariaLabel.includes("send") ||
      ariaLabel.includes("submit") ||
      id.includes("send") ||
      btn.type === "submit"
    ) {
      return true;
    }

    const matchedBtn = findSubmitButton();
    return matchedBtn && (btn === matchedBtn || matchedBtn.contains(target));
  }

  // 2. Intercept Submission Events
  // Capture Enter key on input fields
  document.addEventListener("keydown", async (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      if (e.target.closest("#promptshield-modal, #promptshield-overlay")) {
        return;
      }
      if (Date.now() < bypassUntil) {
        // In bypass window: let event pass through to ChatGPT directly
        return;
      }

      let inputElem = findPromptInput();
      const active = document.activeElement;
      if (active && (active.tagName === "TEXTAREA" || active.getAttribute("contenteditable") === "true")) {
        inputElem = active;
      }
      if (!inputElem) return;

      const isInput = (e.target === inputElem) ||
                      inputElem.contains(e.target) ||
                      !!e.target.closest("#prompt-textarea, [contenteditable='true'], textarea");

      if (isInput) {
        const rawText = getPromptText(inputElem).trim();
        if (rawText.length < 2) return;

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
    if (isSendButtonClicked(e.target)) {
      if (Date.now() < bypassUntil) {
        // In bypass window: let event pass through to ChatGPT directly
        return;
      }

      const inputElem = findPromptInput();
      if (!inputElem) return;

      const rawText = getPromptText(inputElem).trim();
      if (rawText.length < 2) return;

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

    // Check if extension context is valid
    if (!isExtensionContextValid()) {
      isIntercepting = false;
      console.warn("[PromptShield] Extension context invalidated (extension was reloaded). Please refresh page (F5).");
      showMiniStatus("⚠️ Extension reloaded — please refresh page (F5)");
      setTimeout(hideMiniStatus, 3500);
      triggerDirectSubmit(promptText, inputElem);
      return;
    }

    showMiniStatus("PromptShield Analyzing...");

    // Safety timeout: if analysis takes longer than 7 seconds, release to prevent freezing
    let completed = false;
    const safetyTimer = setTimeout(() => {
      if (!completed) {
        completed = true;
        isIntercepting = false;
        hideMiniStatus();
        console.warn("[PromptShield] Analysis timeout. Releasing prompt directly.");
        triggerDirectSubmit(promptText, inputElem);
      }
    }, 7000);

    safeSendMessage(
      { type: "ANALYZE_PROMPT", payload: { prompt: promptText } },
      (analyzeResponse) => {
        if (completed) return;
        completed = true;
        clearTimeout(safetyTimer);
        isIntercepting = false;
        hideMiniStatus();

        if (!analyzeResponse || !analyzeResponse.success) {
          const err = analyzeResponse?.error || "offline";
          console.warn("[PromptShield] Analysis error or offline:", err);
          if (err === "EXTENSION_CONTEXT_INVALIDATED" || (typeof err === "string" && (err.includes("context invalidated") || err.includes("reloaded")))) {
            showMiniStatus("⚠️ Extension reloaded — please refresh page (F5)");
          } else {
            showMiniStatus("⚠️ PromptShield offline (start python run_server.py)");
          }
          setTimeout(hideMiniStatus, 3500);
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        if (analyzeResponse.bypassed) {
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        const data = analyzeResponse.data;
        const entities = data.detected_entities || [];
        const policies = data.entity_policies || [];
        const needsProtection = policies.some(
          p => p.decision === "MASK" || p.decision === "USER_APPROVAL"
        );

        // If no entities require masking or user approval, pass directly through!
        if (!needsProtection || entities.length === 0) {
          console.log("[PromptShield] No entities require protection. Passing prompt through.");
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        // Sensitive entities found! Display Security Modal
        console.log(`[PromptShield] Intercepted prompt with ${entities.length} entities (${policies.filter(p => p.decision === 'MASK').length} to mask).`);
        renderSecurityModal(data, promptText, inputElem);
      }
    );
  }

  function triggerDirectSubmit(text, inputElem) {
    bypassUntil = Date.now() + 6000;
    const elem = (inputElem && inputElem.isConnected) ? inputElem : (findPromptInput() || lastActiveInput);
    if (!elem) {
      console.warn("[PromptShield] Could not locate prompt input element.");
      return;
    }

    setPromptText(elem, text);

    let submitted = false;

    const dispatchSubmission = () => {
      if (submitted) return;

      const currentInput = (elem && elem.isConnected) ? elem : (findPromptInput() || lastActiveInput);
      if (!currentInput) return;

      currentInput.focus();
      const sendBtn = findSubmitButton(currentInput);

      // Strategy: Click send button if found, otherwise dispatch Enter key on the input
      if (sendBtn) {
        try {
          sendBtn.disabled = false;
          sendBtn.removeAttribute("disabled");
          sendBtn.removeAttribute("aria-disabled");

          const evtOpts = { bubbles: true, cancelable: true, composed: true, view: window };
          sendBtn.dispatchEvent(new PointerEvent("pointerdown", evtOpts));
          sendBtn.dispatchEvent(new MouseEvent("mousedown", evtOpts));
          sendBtn.dispatchEvent(new PointerEvent("pointerup", evtOpts));
          sendBtn.dispatchEvent(new MouseEvent("mouseup", evtOpts));
          sendBtn.click();
          submitted = true;
          console.log("[PromptShield] Clicked send button.");
          return;
        } catch (err) {
          console.warn("[PromptShield] Error clicking send button:", err);
        }
      }

      // If no send button found, dispatch Enter keydown on the input element
      const kOpts = {
        key: "Enter",
        code: "Enter",
        keyCode: 13,
        which: 13,
        bubbles: true,
        cancelable: true,
        composed: true,
        view: window,
      };
      currentInput.dispatchEvent(new KeyboardEvent("keydown", kOpts));
      currentInput.dispatchEvent(new KeyboardEvent("keypress", kOpts));
      currentInput.dispatchEvent(new KeyboardEvent("keyup", kOpts));
      submitted = true;
      console.log("[PromptShield] Dispatched Enter keyboard events to input.");
    };

    // Initial dispatch (80ms)
    setTimeout(() => {
      dispatchSubmission();
    }, 80);

    // Follow-up verification (300ms): if not submitted and LLM hasn't started generating, retry
    setTimeout(() => {
      if (!isLLMGenerating()) {
        const remaining = getPromptText(elem).trim();
        if (remaining.length > 0 && !submitted) {
          console.log("[PromptShield] Retrying submission...");
          dispatchSubmission();
        }
      }
    }, 300);
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

    const hasInsufficientContext = Boolean(analysisData.has_insufficient_context);
    const firstEntityText = policies[0]?.entity_text || "this entity";

    // Track user approved entity IDs for release as literals, and user forced mask IDs
    const approvedEntityIds = new Set();
    const maskedEntityIds = new Set();

    policies.forEach((p) => {
      const eid = p.entity_id || `${p.entity_type}:${p.entity_text}`;
      if (p.decision === "RETAIN") {
        approvedEntityIds.add(eid);
      } else {
        maskedEntityIds.add(eid);
      }
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
          ${hasInsufficientContext ? `
            <div class="ps-context-alert">
              <div class="ps-context-alert-header">
                <span class="ps-context-alert-icon">⚠️</span>
                <span class="ps-context-alert-title">Prompt Lacks Operational Context</span>
              </div>
              <p class="ps-context-alert-msg">
                ${escapeHtml(analysisData.context_warning || "This prompt contains an isolated entity with no action verbs. PromptShield cannot determine your operational intent. Please provide more context so the AI can assist properly without exposing naked identities.")}
              </p>
              <div class="ps-context-suggestions">
                <span class="ps-context-sugg-title">💡 Recommended: Click a template to enter a prompt with context:</span>
                <div class="ps-sugg-chip-group">
                  <button type="button" class="ps-sugg-chip" data-template="Draft a brief email to ${escapeHtml(firstEntityText)} regarding project updates:">
                    Draft an email to ${escapeHtml(firstEntityText)}...
                  </button>
                  <button type="button" class="ps-sugg-chip" data-template="Summarize the background and project role of ${escapeHtml(firstEntityText)}:">
                    Summarize role of ${escapeHtml(firstEntityText)}...
                  </button>
                  <button type="button" class="ps-sugg-chip" data-template="Provide details and next steps regarding ${escapeHtml(firstEntityText)}:">
                    Provide details on ${escapeHtml(firstEntityText)}...
                  </button>
                </div>
              </div>
            </div>
          ` : ""}

          <div>
            <div class="ps-section-heading">Prompt Privacy Preview</div>
            <div class="ps-prompt-preview">${highlightedPrompt}</div>
          </div>

          <div>
            <div class="ps-section-heading">Detected Sensitive Data & Policy Actions (${policies.length})</div>
            <div class="ps-entities-list">
              ${policies.map((p, idx) => {
      const isCredential = p.entity_type === "PASSWORD" || p.entity_type === "API_KEY" || p.entity_type === "ACCESS_TOKEN";
      const isUserApproval = p.decision === "USER_APPROVAL";
      const eid = p.entity_id || `${p.entity_type}:${p.entity_text}`;

      let badgeClass = "ps-badge-mask";
      let badgeText = "🔒 MASK TOKEN";
      if (isUserApproval) {
        badgeClass = "ps-badge-approval";
        badgeText = "⚠️ APPROVAL NEEDED";
      } else if (p.decision === "RETAIN") {
        badgeClass = "ps-badge-retain";
        badgeText = "✓ RETAIN CONTEXT";
      }

      return `
                  <div class="ps-entity-row" data-eid="${escapeHtml(eid)}">
                    <div class="ps-entity-meta">
                      <span class="ps-entity-name">${escapeHtml(p.entity_text)}</span>
                      <span class="ps-entity-sub">${p.entity_type} • ${escapeHtml(p.rationale || "Evaluated by policy engine")}</span>
                    </div>

                    ${isCredential ? `
                      <span class="ps-entity-decision-badge ps-badge-mask">🔒 MANDATORY MASK</span>
                    ` : `
                      <div class="ps-approval-actions">
                        <span class="ps-entity-decision-badge ${badgeClass}" id="psBadge-${idx}">${badgeText}</span>
                        <button type="button" class="ps-toggle-btn ps-toggle-btn-mask ${p.decision !== 'RETAIN' ? 'active' : ''}" data-idx="${idx}" data-eid="${escapeHtml(eid)}">
                          Mask
                        </button>
                        <button type="button" class="ps-toggle-btn ps-toggle-btn-approve ${p.decision === 'RETAIN' ? 'active' : ''}" data-idx="${idx}" data-eid="${escapeHtml(eid)}">
                          Allow
                        </button>
                      </div>
                    `}
                  </div>
                `;
    }).join("")}
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="ps-modal-footer">
          <button type="button" id="psBtnBypass" class="ps-btn ps-btn-danger-ghost">Bypass & Send Raw</button>
          <div class="ps-btn-group-right">
            <button type="button" id="psBtnCancel" class="ps-btn ps-btn-ghost">${hasInsufficientContext ? "✏️ Add Context" : "Cancel & Edit"}</button>
            <button type="button" id="psBtnSanitize" class="ps-btn ps-btn-primary">
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

    // Suggestion Chips Click Event (Quickly enrich prompt with context)
    overlay.querySelectorAll(".ps-sugg-chip").forEach((chip) => {
      chip.addEventListener("click", () => {
        const template = chip.getAttribute("data-template");
        overlay.remove();
        const targetInput = (inputElem && inputElem.isConnected) ? inputElem : findPromptInput();
        if (targetInput) {
          setPromptText(targetInput, template);
          targetInput.focus();
        }
      });
    });

    // Update live Prompt Privacy Preview in modal
    function updatePromptPreview() {
      let previewText = escapeHtml(originalPrompt);
      policies.forEach((p) => {
        const eid = p.entity_id || `${p.entity_type}:${p.entity_text}`;
        const isMasked = maskedEntityIds.has(eid);
        const cls = isMasked ? "ps-highlight-mask" : "ps-highlight-retain";
        const regex = new RegExp(`\\b${escapeRegExp(escapeHtml(p.entity_text))}\\b`, "gi");
        previewText = previewText.replace(regex, `<span class="${cls}">$&</span>`);
      });
      const previewBox = overlay.querySelector(".ps-prompt-preview");
      if (previewBox) {
        previewBox.innerHTML = previewText;
      }
    }

    // Interactive Approval / Mask Toggle Buttons
    overlay.querySelectorAll(".ps-toggle-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const idx = btn.getAttribute("data-idx");
        const eid = btn.getAttribute("data-eid");
        const badge = document.getElementById(`psBadge-${idx}`);
        const parentRow = btn.closest(".ps-approval-actions");
        const maskBtn = parentRow.querySelector(".ps-toggle-btn-mask");
        const approveBtn = parentRow.querySelector(".ps-toggle-btn-approve");

        if (btn.classList.contains("ps-toggle-btn-approve")) {
          // User approved this entity! Release as literal
          approvedEntityIds.add(eid);
          maskedEntityIds.delete(eid);
          approveBtn.classList.add("active");
          maskBtn.classList.remove("active");
          if (badge) {
            badge.className = "ps-entity-decision-badge ps-badge-retain";
            badge.textContent = "✓ USER APPROVED (RETAIN)";
          }
        } else {
          // User wants entity masked as token
          approvedEntityIds.delete(eid);
          maskedEntityIds.add(eid);
          maskBtn.classList.add("active");
          approveBtn.classList.remove("active");
          if (badge) {
            badge.className = "ps-entity-decision-badge ps-badge-mask";
            badge.textContent = "🔒 MASK TOKEN";
          }
        }
        updatePromptPreview();
      });
    });

    // Event Listeners for Modal Buttons
    document.getElementById("psBtnCancel").addEventListener("click", () => {
      overlay.remove();
      const targetInput = (inputElem && inputElem.isConnected) ? inputElem : (findPromptInput() || lastActiveInput);
      if (targetInput) targetInput.focus();
    });

    document.getElementById("psBtnBypass").addEventListener("click", () => {
      overlay.remove();
      const targetInput = (inputElem && inputElem.isConnected) ? inputElem : (findPromptInput() || lastActiveInput);
      triggerDirectSubmit(originalPrompt, targetInput);
    });

    document.getElementById("psBtnSanitize").addEventListener("click", () => {
      // Capture targetInput immediately before removing modal overlay
      const targetInput = (inputElem && inputElem.isConnected) ? inputElem : (findPromptInput() || lastActiveInput);

      // 1. Remove modal overlay IMMEDIATELY so page is never frozen
      overlay.remove();
      showMiniStatus("🛡️ PromptShield: Applying Sanitization...");

      safeSendMessage(
        {
          type: "SANITIZE_PROMPT",
          payload: {
            prompt: originalPrompt,
            approved_entity_ids: Array.from(approvedEntityIds),
            masked_entity_ids: Array.from(maskedEntityIds),
          },
        },
        (sanitizeRes) => {
          hideMiniStatus();

          if (!sanitizeRes || !sanitizeRes.success) {
            console.warn("[PromptShield] Sanitization warning:", sanitizeRes?.error);
            showMiniStatus("⚠️ Extension reloaded — please press F5 on ChatGPT");
            setTimeout(hideMiniStatus, 3500);
            return;
          }

          if (sanitizeRes.data) {
            lastSessionId = sanitizeRes.data.session_id;
            const maskedText = sanitizeRes.data.sanitized_prompt;
            console.log("[PromptShield] Injected sanitized prompt into LLM:", maskedText);
            showMiniStatus("✓ Prompt Sanitized & Protected!");
            setTimeout(hideMiniStatus, 2000);
            triggerDirectSubmit(maskedText, targetInput);
          }
        }
      );
    });
  }

  // 5. Response Restoration Engine (Phase 5)
  // Scans incoming LLM responses for {{TYPE_N}} or <TYPE_N> placeholders and restores authorized secrets
  const inFlightRestoreNodes = new WeakSet();
  const fullyRestoredNodes = new WeakSet();
  const PLACEHOLDER_CHECK_RE = /(?:\{\{|\{|\<|\[)([A-Z_]+)_(\d+)(?:\}\}|\}|\>|\])/;

  // Restore lastSessionId from storage on startup
  chrome.storage.local.get(["lastSessionId"], (res) => {
    if (res && res.lastSessionId) {
      lastSessionId = res.lastSessionId;
      console.log("[PromptShield] Loaded active session ID from storage:", lastSessionId);
      setTimeout(scanAndRestoreResponses, 600);
    }
  });

  function isLLMGenerating() {
    // 1. Check for active streaming class on message turns in ChatGPT / Claude / Gemini
    if (document.querySelector(".result-streaming, .streaming, [data-is-streaming='true'], .font-claude-message.streaming")) {
      return true;
    }

    // 2. Check stop buttons (specific to LLM generation, avoiding audio/voice controls)
    const stopBtn = document.querySelector(
      "button[data-testid='stop-button'], button[aria-label='Stop streaming'], button[aria-label='Stop generating'], button[aria-label='Stop Streaming'], button[aria-label='Stop Generating']"
    );
    return !!(
      stopBtn &&
      stopBtn.offsetParent !== null &&
      !stopBtn.disabled &&
      window.getComputedStyle(stopBtn).display !== "none"
    );
  }

  function hasUnrestoredPlaceholders(node) {
    const target = node.closest("article, [data-testid^='conversation-turn-'], main") || node;
    let rawText = target.textContent || "";
    if (target.querySelectorAll) {
      target.querySelectorAll("input, textarea").forEach((inp) => {
        if (inp.value && !inp.closest("#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast")) {
          rawText += " " + inp.value;
        }
      });
      // Also check for custom HTML tags like <LOCATION_1>, <EMAIL_1> created by browser parser
      target.querySelectorAll("*").forEach((el) => {
        if (/^[A-Z_]+_\d+$/i.test(el.tagName)) {
          rawText += ` {{${el.tagName.toUpperCase()}}}`;
        }
      });
    }
    if (PLACEHOLDER_CHECK_RE.test(rawText)) {
      const allPhs = Array.from(rawText.matchAll(/(?:\{\{|\{|\<|\[)([A-Z_]+)_(\d+)(?:\}\}|\}|\>|\])/g)).map((m) => m[0]);
      return allPhs.some((ph) => {
        const clean = ph.replace(/[\{\}\<\>\[\]]/g, "");
        return !(node._psBlockedPhs && (node._psBlockedPhs.has(ph) || node._psBlockedPhs.has(clean) || node._psBlockedPhs.has(`{{${clean}}}`)));
      });
    }
    return false;
  }

  function findAssistantCandidateNodes() {
    const matched = new Set();
    const root = document.querySelector("main") || document.body;

    // 1. Target both assistant messages AND user prompt messages in the chat
    const selectors = [
      // Assistant turns
      "div[data-message-author-role='assistant']",
      "article [data-message-author-role='assistant']",
      ".font-claude-message",
      "message-content",
      "div.prose",
      "div.markdown",
      ".agent-turn",
      // User prompt turns (restore prompt turn after response)
      "div[data-message-author-role='user']",
      "article [data-message-author-role='user']",
      "[data-testid*='user-turn']",
      "[aria-label*='You said' i]",
      ".user-message",
      // General conversation turns
      "[data-testid^='conversation-turn-']",
      "article"
    ];

    document.querySelectorAll(selectors.join(", ")).forEach((el) => {
      if (
        !el.closest(
          "#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast"
        ) &&
        hasUnrestoredPlaceholders(el)
      ) {
        matched.add(el);
      }
    });

    // 2. Scan all text nodes in main to catch prompt turns, response cards, headers, email previews, or widgets
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);
    let textNode;
    while ((textNode = walker.nextNode())) {
      const parent = textNode.parentElement;
      if (!parent) continue;
      if (
        parent.closest(
          "#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast"
        )
      ) {
        continue;
      }
      if (PLACEHOLDER_CHECK_RE.test(textNode.nodeValue || "")) {
        const container = parent.closest(
          "article, [data-message-author-role='assistant'], [data-message-author-role='user'], [data-testid^='conversation-turn-'], div[class*='rounded'], div.markdown"
        ) || parent;
        if (hasUnrestoredPlaceholders(container)) {
          matched.add(container);
        }
      }
    }

    // 3. Scan inputs and textareas in main that may contain placeholders in cards
    root.querySelectorAll("input, textarea").forEach((inp) => {
      if (
        inp.closest(
          "#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast"
        )
      ) {
        return;
      }
      if (inp.value && PLACEHOLDER_CHECK_RE.test(inp.value)) {
        const container = inp.closest(
          "article, [data-message-author-role='assistant'], [data-message-author-role='user'], [data-testid^='conversation-turn-'], div[class*='rounded']"
        ) || inp;
        if (hasUnrestoredPlaceholders(container)) {
          matched.add(container);
        }
      }
    });

    // 4. Scan for custom tag elements (e.g. <LOCATION_1>, <EMAIL_1>) in main
    root.querySelectorAll("*").forEach((el) => {
      if (
        !el.closest(
          "#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast"
        ) &&
        /^[A-Z_]+_\d+$/i.test(el.tagName)
      ) {
        const container = el.closest(
          "article, [data-message-author-role='assistant'], [data-message-author-role='user'], [data-testid^='conversation-turn-'], main"
        ) || el.parentElement;
        if (container && hasUnrestoredPlaceholders(container)) {
          matched.add(container);
        }
      }
    });

    return Array.from(matched);
  }

  function scanAndRestoreResponses() {
    if (isLLMGenerating()) {
      return;
    }

    const candidateNodes = findAssistantCandidateNodes();
    candidateNodes.forEach((node) => {
      if (hasUnrestoredPlaceholders(node)) {
        fullyRestoredNodes.delete(node);
      } else {
        fullyRestoredNodes.add(node);
      }

      if (inFlightRestoreNodes.has(node) || fullyRestoredNodes.has(node)) return;

      clearTimeout(node._psRestoreDebounce);
      node._psRestoreDebounce = setTimeout(() => {
        if (!isLLMGenerating()) {
          performResponseRestoration(node);
        }
      }, 300);
    });
  }

  function performResponseRestoration(node) {
    if (inFlightRestoreNodes.has(node)) return;
    if (isLLMGenerating()) return;

    if (!hasUnrestoredPlaceholders(node)) {
      fullyRestoredNodes.add(node);
      return;
    }

    // Gather comprehensive text across the turn container and any associated card widgets
    const turn = node.closest("article, [data-testid^='conversation-turn-'], main") || node;
    let rawText = turn.textContent || node.textContent || "";
    if (turn.querySelectorAll) {
      turn.querySelectorAll("input, textarea").forEach((inp) => {
        if (inp.value && !inp.closest("#prompt-textarea, #promptshield-overlay, #promptshield-modal, #promptshield-restoration-toast")) {
          rawText += " " + inp.value;
        }
      });
      // Capture any custom HTML tag elements (e.g. <location_1></location_1>)
      turn.querySelectorAll("*").forEach((el) => {
        if (/^[A-Z_]+_\d+$/i.test(el.tagName)) {
          rawText += ` {{${el.tagName.toUpperCase()}}}`;
        }
      });
    }

    inFlightRestoreNodes.add(node);
    console.log("[PromptShield] Calling Phase 5 /restore for turn. Session ID:", lastSessionId);

    safeSendMessage(
      {
        type: "RESTORE_RESPONSE",
        payload: {
          llm_response: rawText,
          session_id: lastSessionId || null,
          allow_all_restoration: false,
        },
      },
      (res) => {
        inFlightRestoreNodes.delete(node);

        if (!res || !res.success || !res.data) {
          console.warn("[PromptShield] Restoration response error:", res?.error);
          return;
        }

        const data = res.data;
        if (data.blocked_placeholders && data.blocked_placeholders.length > 0) {
          if (!node._psBlockedPhs) node._psBlockedPhs = new Set();
          data.blocked_placeholders.forEach((ph) => {
            node._psBlockedPhs.add(ph);
            const clean = ph.replace(/[\{\}\<\>\[\]]/g, "");
            node._psBlockedPhs.add(clean);
            node._psBlockedPhs.add(`{{${clean}}}`);
          });
        }

        if (data.restored_count > 0 && data.restored_map) {
          const restoredMap = data.restored_map || {};

          // Safely replace text nodes, inputs, attributes, and custom tag elements in this turn
          const targetScope = node.closest("article, [data-testid^='conversation-turn-'], main") || document.body;
          safelyReplacePlaceholders(targetScope, restoredMap);

          // Also execute a sweep across main to ensure user prompt turns & sibling cards are 100% restored
          const mainScope = document.querySelector("main") || document.body;
          if (mainScope !== targetScope) {
            safelyReplacePlaceholders(mainScope, restoredMap);
          }

          // Display non-intrusive floating toast in corner
          showRestorationToast(data.restored_count);
          console.log(`[PromptShield] Successfully restored ${data.restored_count} tokens in conversation:`, restoredMap);
        }

        // If no more unblocked placeholders remain, mark node as fully restored
        if (!hasUnrestoredPlaceholders(node)) {
          fullyRestoredNodes.add(node);
        }
      }
    );
  }

  function setNativeInputValue(element, value) {
    try {
      const isTextarea = element.tagName === "TEXTAREA";
      const proto = isTextarea ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, "value")?.set;
      if (setter) {
        setter.call(element, value);
      } else {
        element.value = value;
      }
    } catch (_) {
      element.value = value;
    }
    try {
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    } catch (_) {}
  }

  function safelyReplacePlaceholders(rootElem, restoredMap) {
    if (!restoredMap || Object.keys(restoredMap).length === 0 || !rootElem) return;

    // 1. Safe text node walker replacement (preserves React virtual DOM and event listeners)
    // Exclude only the active compose input (#prompt-textarea) and PromptShield's own modal UI
    const walker = document.createTreeWalker(rootElem, NodeFilter.SHOW_TEXT, null, false);
    let textNode;
    const nodesToUpdate = [];
    while ((textNode = walker.nextNode())) {
      if (
        textNode.parentElement &&
        textNode.parentElement.closest(
          "#promptshield-restoration-toast, #promptshield-overlay, #promptshield-modal, #prompt-textarea"
        )
      ) {
        continue;
      }
      for (const ph of Object.keys(restoredMap)) {
        if (textNode.nodeValue && textNode.nodeValue.includes(ph)) {
          nodesToUpdate.push(textNode);
          break;
        }
      }
    }

    for (const tn of nodesToUpdate) {
      let val = tn.nodeValue;
      for (const [ph, secret] of Object.entries(restoredMap)) {
        val = val.replaceAll(ph, secret);
      }
      tn.nodeValue = val;
    }

    // 2. Safe input / textarea value updates with native React prototype setter
    if (rootElem.querySelectorAll) {
      rootElem.querySelectorAll("input, textarea").forEach((inp) => {
        if (inp.closest("#promptshield-restoration-toast, #promptshield-overlay, #promptshield-modal, #prompt-textarea")) {
          return;
        }
        try {
          if (inp.value && typeof inp.value === "string") {
            let val = inp.value;
            let modified = false;
            for (const [ph, secret] of Object.entries(restoredMap)) {
              if (val.includes(ph)) {
                val = val.replaceAll(ph, secret);
                modified = true;
              }
            }
            if (modified) {
              setNativeInputValue(inp, val);
            }
          }
        } catch (_) {}
      });

      // 3. Update any element attributes (e.g. placeholder, title, aria-label)
      const attributeNames = ["placeholder", "title", "aria-label"];
      rootElem.querySelectorAll("*").forEach((el) => {
        if (el.closest("#promptshield-restoration-toast, #promptshield-overlay, #promptshield-modal, #prompt-textarea")) {
          return;
        }
        for (const attr of attributeNames) {
          if (el.hasAttribute(attr)) {
            let attrVal = el.getAttribute(attr);
            if (attrVal) {
              let updated = false;
              for (const [ph, secret] of Object.entries(restoredMap)) {
                if (attrVal.includes(ph)) {
                  attrVal = attrVal.replaceAll(ph, secret);
                  updated = true;
                }
              }
              if (updated) {
                el.setAttribute(attr, attrVal);
              }
            }
          }
        }
      });

      // 4. Safe replacement for custom HTML tag elements created by browser (e.g. <LOCATION_1></LOCATION_1>)
      rootElem.querySelectorAll("*").forEach((el) => {
        if (
          el.closest(
            "#promptshield-restoration-toast, #promptshield-overlay, #promptshield-modal, #prompt-textarea"
          )
        ) {
          return;
        }
        const tag = el.tagName.toUpperCase();
        for (const [ph, secret] of Object.entries(restoredMap)) {
          const cleanPh = ph.replace(/[\{\}\<\>\[\]]/g, "").toUpperCase();
          if (tag === cleanPh && secret) {
            const textNode = document.createTextNode(secret);
            el.replaceWith(textNode);
            break;
          }
        }
      });
    }
  }

  function showRestorationToast(count) {
    let toast = document.getElementById("promptshield-restoration-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "promptshield-restoration-toast";
      toast.style.cssText = `
        position: fixed !important;
        bottom: 24px !important;
        right: 24px !important;
        z-index: 99999999 !important;
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
        color: #34d399 !important;
        padding: 8px 16px !important;
        border-radius: 999px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        backdrop-filter: blur(8px) !important;
        animation: psSlideUp 0.3s ease-out !important;
        transition: opacity 0.5s ease !important;
      `;
      document.body.appendChild(toast);
    }
    toast.innerHTML = `
      <span style="font-size: 14px;">🛡️</span>
      <span>PromptShield Restored <strong>${count}</strong> confidential token${count > 1 ? 's' : ''}</span>
    `;
    toast.style.opacity = "1";
    toast.style.display = "flex";

    clearTimeout(toast._dismissTimer);
    toast._dismissTimer = setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => {
        toast.style.display = "none";
      }, 500);
    }, 4000);
  }

  // Debounced DOM observer that triggers when responses arrive
  let restoreDebounceTimer = null;
  const responseObserver = new MutationObserver(() => {
    if (isLLMGenerating()) return;
    clearTimeout(restoreDebounceTimer);
    restoreDebounceTimer = setTimeout(() => {
      scanAndRestoreResponses();
    }, 400);
  });

  responseObserver.observe(document.body, { childList: true, subtree: true });

  // Periodic safety scanner ensuring completed responses are restored even if mutation debounce was skipped
  setInterval(() => {
    scanAndRestoreResponses();
  }, 1000);

  // 6. Floating Status Pill Helper
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

