/**
 * PromptShield AI — Popup UI Controller
 * Manages user preferences, server health checks, and in-popup quick sandbox.
 */

document.addEventListener("DOMContentLoaded", async () => {
  // DOM Elements
  const masterToggle = document.getElementById("masterToggle");
  const backendStatus = document.getElementById("backendStatus");
  const statusText = document.getElementById("statusText");
  const levelBtns = document.querySelectorAll(".level-btn");
  const statScanned = document.getElementById("statScanned");
  const statProtected = document.getElementById("statProtected");
  const statRisks = document.getElementById("statRisks");
  const testInput = document.getElementById("testInput");
  const testScanBtn = document.getElementById("testScanBtn");
  const testClearBtn = document.getElementById("testClearBtn");
  const testOutputArea = document.getElementById("testOutputArea");
  const testRiskBadge = document.getElementById("testRiskBadge");
  const testLatency = document.getElementById("testLatency");
  const testSanitizedText = document.getElementById("testSanitizedText");
  const testDecisionsList = document.getElementById("testDecisionsList");

  // 1. Load Current Configuration from Background
  chrome.runtime.sendMessage({ type: "GET_CONFIG" }, (response) => {
    if (response && response.success && response.data) {
      const cfg = response.data;
      masterToggle.checked = cfg.enabled !== false;
      updateLevelUI(cfg.securityLevel || "BALANCED");
      updateStatsUI(cfg.stats);
    }
  });

  // 2. Perform Backend Health Check
  checkBackendHealth();

  function checkBackendHealth() {
    backendStatus.className = "status-pill status-checking";
    statusText.textContent = "Checking...";

    chrome.runtime.sendMessage({ type: "CHECK_HEALTH" }, (response) => {
      if (response && response.success) {
        backendStatus.className = "status-pill status-online";
        statusText.textContent = "Backend Active";
      } else {
        backendStatus.className = "status-pill status-offline";
        statusText.textContent = "Offline (start server)";
        backendStatus.title = "Ensure 'python run_server.py' is running on port 8000";
      }
    });
  }

  // 3. Master Toggle Handler
  masterToggle.addEventListener("change", (e) => {
    const enabled = e.target.checked;
    chrome.runtime.sendMessage({
      type: "SET_CONFIG",
      payload: { enabled },
    });
  });

  // 4. Security Level Selection
  levelBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const level = btn.getAttribute("data-level");
      updateLevelUI(level);
      chrome.runtime.sendMessage({
        type: "SET_CONFIG",
        payload: { securityLevel: level },
      });
    });
  });

  function updateLevelUI(activeLevel) {
    levelBtns.forEach((b) => {
      if (b.getAttribute("data-level") === activeLevel) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });
  }

  function updateStatsUI(stats) {
    if (!stats) return;
    statScanned.textContent = stats.promptsScanned || 0;
    statProtected.textContent = stats.entitiesProtected || 0;
    statRisks.textContent = stats.risksAverted || 0;
  }

  // 5. In-Popup Quick Tester (Live Sandbox)
  testScanBtn.addEventListener("click", async () => {
    const text = testInput.value.trim();
    if (!text) {
      testInput.focus();
      return;
    }

    testScanBtn.disabled = true;
    testScanBtn.textContent = "Scanning...";
    const startTime = performance.now();

    // Call Analyze & Sanitize in parallel
    chrome.runtime.sendMessage(
      { type: "ANALYZE_PROMPT", payload: { prompt: text } },
      (analyzeRes) => {
        chrome.runtime.sendMessage(
          { type: "SANITIZE_PROMPT", payload: { prompt: text } },
          (sanitizeRes) => {
            testScanBtn.disabled = false;
            testScanBtn.innerHTML = `
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
              Analyze & Mask
            `;

            const latency = Math.round(performance.now() - startTime);
            renderTestResults(analyzeRes, sanitizeRes, latency);

            // Refresh stats
            chrome.runtime.sendMessage({ type: "GET_CONFIG" }, (r) => {
              if (r && r.data) updateStatsUI(r.data.stats);
            });
          }
        );
      }
    );
  });

  testClearBtn.addEventListener("click", () => {
    testInput.value = "";
    testOutputArea.classList.add("hidden");
    testInput.focus();
  });

  function renderTestResults(analyzeRes, sanitizeRes, latency) {
    testOutputArea.classList.remove("hidden");
    testLatency.textContent = `${latency} ms`;

    // 1. Risk Badge
    let riskLevel = "LOW";
    let riskScore = 0;
    let policies = [];

    if (analyzeRes && analyzeRes.success && analyzeRes.data) {
      const raw = analyzeRes.data.overall_risk_score || 0;
      riskScore = raw <= 1.0 ? Math.round(raw * 100) : Math.round(raw);
      policies = analyzeRes.data.entity_policies || [];
      const hasHigh = policies.some(p => p.entity_type === "PASSWORD" || p.entity_type === "API_KEY" || p.entity_type === "CREDIT_CARD");
      if (hasHigh || riskScore >= 70) riskLevel = "HIGH";
      else if (riskScore >= 35 || policies.some(p => p.decision === "MASK")) riskLevel = "MEDIUM";
      else riskLevel = "LOW";
    }

    testRiskBadge.className = `risk-badge risk-${riskLevel.toLowerCase()}`;
    testRiskBadge.textContent = `${riskLevel} RISK (${Math.round(riskScore)}/100)`;

    // 2. Sanitized Prompt
    if (sanitizeRes && sanitizeRes.success && sanitizeRes.data) {
      testSanitizedText.textContent = sanitizeRes.data.sanitized_prompt;
    } else {
      testSanitizedText.textContent = "(Could not sanitize prompt — check backend server)";
    }

    // 3. Decisions List
    testDecisionsList.innerHTML = "";
    if (policies.length === 0) {
      testDecisionsList.innerHTML = `<span style="color: #94a3b8; font-size: 11px;">No sensitive entities detected — safe to transmit.</span>`;
    } else {
      policies.forEach((p) => {
        const item = document.createElement("div");
        item.className = "decision-item";

        let pillClass = "pill-review";
        if (p.decision === "MASK") pillClass = "pill-mask";
        else if (p.decision === "RETAIN") pillClass = "pill-retain";

        item.innerHTML = `
          <span><strong>${escapeHtml(p.entity_text)}</strong> (${p.entity_type})</span>
          <span class="decision-pill ${pillClass}">${p.decision}</span>
        `;
        testDecisionsList.appendChild(item);
      });
    }
  }

  function escapeHtml(str) {
    return str.replace(/[&<>'"]/g, 
      tag => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        "'": '&#39;',
        '"': '&quot;'
      }[tag] || tag)
    );
  }
});
