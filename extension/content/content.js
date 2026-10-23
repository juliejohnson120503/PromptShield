/**
 * PromptShield AI — Content Script (Manifest V3)
 * Real-time prompt interceptor for ChatGPT, Claude, Gemini, and LLM web interfaces.
 */

(() => {
  let isIntercepting = false;
  let bypassNextSubmit = false;
  let lastActiveInput = null;
  let lastSessionId = null;

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
      "button[data-testid='send-button'], button[data-testid*='send' i], button[aria-label*='Send' i], button[data-testid='composer-send-button'], button[data-testid='fruitjuice-send-button'], form button[type='submit']"
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

  function isSendButtonClicked(target) {
    if (!target) return false;
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
      let inputElem = findPromptInput();
      const active = document.activeElement;
      if (active && (active.tagName === "TEXTAREA" || active.getAttribute("contenteditable") === "true")) {
        inputElem = active;
      }
      if (!inputElem) return;

      // Check if event targets our input or child
      const isInput = (e.target === inputElem) ||
                      inputElem.contains(e.target) ||
                      !!e.target.closest("#prompt-textarea, [contenteditable='true'], textarea");

      if (isInput) {
        if (bypassNextSubmit) {
          bypassNextSubmit = false;
          return;
        }

        const rawText = getPromptText(inputElem).trim();
        if (rawText.length < 2) return; // Skip trivial inputs

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
      if (bypassNextSubmit) {
        bypassNextSubmit = false;
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

    showMiniStatus("PromptShield Analyzing...");

    chrome.runtime.sendMessage(
      { type: "ANALYZE_PROMPT", payload: { prompt: promptText } },
      (analyzeResponse) => {
        isIntercepting = false;
        hideMiniStatus();

        if (chrome.runtime.lastError) {
          console.warn("[PromptShield] Extension context error:", chrome.runtime.lastError.message);
          showMiniStatus("⚠️ Extension reloaded — please refresh page (F5)");
          setTimeout(hideMiniStatus, 3500);
          triggerDirectSubmit(promptText, inputElem);
          return;
        }

        if (!analyzeResponse || !analyzeResponse.success) {
          console.warn("[PromptShield] Analysis error or offline:", analyzeResponse?.error);
          showMiniStatus("⚠️ PromptShield offline (start python run_server.py)");
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

    // Poll reactively for React/ProseMirror to update state and enable the send button
    let attempts = 0;
    const maxAttempts = 8;
    const pollInterval = setInterval(() => {
      attempts++;
      const sendBtn = findSubmitButton();
      const isEnabled = sendBtn && !sendBtn.disabled && sendBtn.getAttribute("aria-disabled") !== "true";

      if (isEnabled) {
        clearInterval(pollInterval);
        console.log("[PromptShield] Send button enabled, triggering click.");
        sendBtn.click();
        return;
      }

      if (attempts >= maxAttempts) {
        clearInterval(pollInterval);
        console.log("[PromptShield] Attempting fallback submission techniques...");

        // Fallback A: Native form requestSubmit
        const form = elem.closest("form");
        if (form && typeof form.requestSubmit === "function") {
          try {
            form.requestSubmit();
            console.log("[PromptShield] Form requestSubmit dispatched successfully.");
            return;
          } catch (_) { }
        }

        // Fallback B: Mouse/Pointer event sequence on send button
        if (sendBtn) {
          sendBtn.disabled = false;
          sendBtn.removeAttribute("disabled");
          sendBtn.removeAttribute("aria-disabled");
          sendBtn.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true, cancelable: true }));
          sendBtn.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true }));
          sendBtn.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, cancelable: true }));
          sendBtn.dispatchEvent(new MouseEvent("mouseup", { bubbles: true, cancelable: true }));
          sendBtn.click();
        }

        // Fallback C: Keyboard Enter event sequence
        elem.focus();
        elem.dispatchEvent(new KeyboardEvent("keydown", {
          key: "Enter",
          code: "Enter",
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true,
        }));
      }
    }, 50);
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

    // Track user approved entity IDs for release as literals
    const approvedEntityIds = new Set();

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
          <button id="psBtnBypass" class="ps-btn ps-btn-danger-ghost">Bypass & Send Raw</button>
          <div class="ps-btn-group-right">
            <button id="psBtnCancel" class="ps-btn ps-btn-ghost">${hasInsufficientContext ? "✏️ Add Context" : "Cancel & Edit"}</button>
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
          approveBtn.classList.add("active");
          maskBtn.classList.remove("active");
          if (badge) {
            badge.className = "ps-entity-decision-badge ps-badge-retain";
            badge.textContent = "✓ USER APPROVED (RETAIN)";
          }
        } else {
          // User wants entity masked as token
          approvedEntityIds.delete(eid);
          maskBtn.classList.add("active");
          approveBtn.classList.remove("active");
          if (badge) {
            badge.className = "ps-entity-decision-badge ps-badge-mask";
            badge.textContent = "🔒 MASK TOKEN";
          }
        }
      });
    });

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
        {
          type: "SANITIZE_PROMPT",
          payload: {
            prompt: originalPrompt,
            approved_entity_ids: Array.from(approvedEntityIds),
          },
        },
        (sanitizeRes) => {
          overlay.remove();
          const targetInput = (inputElem && inputElem.isConnected) ? inputElem : findPromptInput();
          if (sanitizeRes && sanitizeRes.success && sanitizeRes.data) {
            lastSessionId = sanitizeRes.data.session_id;
            const maskedText = sanitizeRes.data.sanitized_prompt;
            console.log("[PromptShield] Injected sanitized prompt into LLM:", maskedText);
            console.log("[PromptShield] Active session ID for restoration:", lastSessionId);
            triggerDirectSubmit(maskedText, targetInput);
          } else {
            console.error("[PromptShield] Failed to sanitize prompt:", sanitizeRes);
            triggerDirectSubmit(originalPrompt, targetInput);
          }
        }
      );
    });
  }

  // 5. Response Restoration Engine (Phase 5)
  // Scans incoming LLM responses for <TYPE_N> placeholders and restores authorized secrets
  const processedAssistantNodes = new WeakSet();
  const PLACEHOLDER_CHECK_RE = /<([A-Z_]+)_(\d+)>/;

  function isLLMGenerating() {
    return !!document.querySelector(
      "button[data-testid='stop-button'], button[aria-label*='Stop' i], button[aria-label='Stop streaming'], button[aria-label='Stop generating'], .stop-button, [data-testid*='stop' i]"
    );
  }

  function isExcludedNode(el) {
    if (!el || !el.closest) return true;
    return !!el.closest(
      "aside, nav, [role='navigation'], [data-testid*='sidebar' i], .sidebar, " +
      "#prompt-textarea, textarea, [contenteditable='true'], " +
      "#promptshield-overlay, #promptshield-floating-indicator, #promptshield-mini-status, " +
      "#promptshield-restoration-toast, header, [role='banner'], " +
      "[data-message-author-role='user'], [data-testid*='user' i]"
    );
  }

  function findAssistantCandidateNodes() {
    const matched = [];

    // Strictly target markdown content elements inside assistant message containers
    const assistantSelectors = [
      "div[data-message-author-role='assistant'] div.markdown",
      "article div.markdown",
      ".font-claude-message",
      "message-content",
      ".model-response-text",
      ".message.assistant .content",
      ".chat-message.assistant .message-text"
    ];

    const elements = document.querySelectorAll(assistantSelectors.join(", "));
    elements.forEach((el) => {
      // Must not be in any excluded container (sidebar, prompt input, user turn)
      if (isExcludedNode(el)) return;

      // Ensure the turn does not contain user action buttons (e.g. edit prompt)
      const turn = el.closest("article, div[class*='turn']");
      if (turn) {
        if (turn.querySelector("button[aria-label*='Edit' i], [data-message-author-role='user'], [data-testid*='user' i]")) {
          return;
        }
      }

      if (PLACEHOLDER_CHECK_RE.test(el.textContent || "")) {
        matched.push(el);
      }
    });

    return matched;
  }

  function scanAndRestoreResponses() {
    // CRITICAL: Never touch or scan DOM while LLM is actively streaming tokens!
    if (isLLMGenerating()) {
      return;
    }

    // Only restore if we have an active session from a sanitized prompt
    if (!lastSessionId) {
      return;
    }

    const candidateNodes = findAssistantCandidateNodes();

    candidateNodes.forEach((node) => {
      if (processedAssistantNodes.has(node)) return;

      const text = node.textContent || "";
      if (PLACEHOLDER_CHECK_RE.test(text)) {
        clearTimeout(node._psRestoreDebounce);
        node._psRestoreDebounce = setTimeout(() => {
          if (!isLLMGenerating()) {
            performResponseRestoration(node);
          }
        }, 600);
      }
    });
  }

  function performResponseRestoration(node) {
    if (processedAssistantNodes.has(node)) return;
    if (isLLMGenerating()) return;
    if (!lastSessionId) return;

    const rawText = node.textContent || "";
    if (!PLACEHOLDER_CHECK_RE.test(rawText)) return;

    processedAssistantNodes.add(node);
    console.log("[PromptShield] Calling Phase 5 /restore. Local session ID:", lastSessionId);

    chrome.runtime.sendMessage(
      {
        type: "RESTORE_RESPONSE",
        payload: {
          llm_response: rawText,
          session_id: lastSessionId,
          allow_all_restoration: false,
        },
      },
      (res) => {
        if (res && res.success && res.data && res.data.restored_count > 0) {
          const restoredMap = res.data.restored_map || {};

          // Safely replace text nodes without wiping HTML tags, code blocks, or React hierarchy
          safelyReplacePlaceholders(node, restoredMap);

          // Display non-intrusive floating toast in corner (leaving ChatGPT interface 100% native!)
          showRestorationToast(res.data.restored_count);
          console.log(`[PromptShield] Successfully restored ${res.data.restored_count} tokens in response.`);
        }
      }
    );
  }

  function safelyReplacePlaceholders(rootElem, restoredMap) {
    if (!restoredMap || Object.keys(restoredMap).length === 0) return;
    const walker = document.createTreeWalker(rootElem, NodeFilter.SHOW_TEXT, null, false);
    let textNode;
    const nodesToUpdate = [];
    while ((textNode = walker.nextNode())) {
      if (textNode.parentElement && textNode.parentElement.closest("#promptshield-restoration-toast, #promptshield-overlay")) {
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

  // Debounced DOM observer that only triggers when streaming is idle
  let restoreDebounceTimer = null;
  const responseObserver = new MutationObserver(() => {
    if (!lastSessionId) return;
    if (isLLMGenerating()) return;
    clearTimeout(restoreDebounceTimer);
    restoreDebounceTimer = setTimeout(() => {
      scanAndRestoreResponses();
    }, 800);
  });

  responseObserver.observe(document.body, { childList: true, subtree: true });

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

