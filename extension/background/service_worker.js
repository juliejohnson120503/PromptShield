/**
 * PromptShield AI — Manifest V3 Background Service Worker
 * Handles REST API communication with the local PromptShield FastAPI service.
 */

const DEFAULT_CONFIG = {
  enabled: true,
  apiBaseUrl: "http://127.0.0.1:8000",
  securityLevel: "BALANCED", // STRICT, BALANCED, PERMISSIVE
  autoMask: false, // If true, automatically masks without modal; if false, shows interactive approval modal
  stats: {
    promptsScanned: 0,
    promptsMasked: 0,
    entitiesProtected: 0,
    risksAverted: 0,
  },
};

// Initialize default settings on install
chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(null);
  const updated = { ...DEFAULT_CONFIG, ...current };
  if (!updated.stats) updated.stats = DEFAULT_CONFIG.stats;
  await chrome.storage.local.set(updated);
  updateBadge(updated.enabled);
  console.log("[PromptShield] Background Service Worker initialized with config:", updated);
});

// Update extension icon badge
function updateBadge(enabled) {
  if (enabled) {
    chrome.action.setBadgeText({ text: "ON" });
    chrome.action.setBadgeBackgroundColor({ color: "#10b981" }); // Emerald green
  } else {
    chrome.action.setBadgeText({ text: "OFF" });
    chrome.action.setBadgeBackgroundColor({ color: "#64748b" }); // Slate gray
  }
}

// Listen for messages from content scripts and popup UI
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const { type, payload } = message;

  (async () => {
    try {
      const config = await chrome.storage.local.get(null);
      const baseUrl = config.apiBaseUrl || DEFAULT_CONFIG.apiBaseUrl;

      switch (type) {
        case "GET_CONFIG": {
          sendResponse({ success: true, data: config });
          break;
        }

        case "SET_CONFIG": {
          await chrome.storage.local.set(payload);
          if (typeof payload.enabled === "boolean") {
            updateBadge(payload.enabled);
          }
          sendResponse({ success: true });
          break;
        }

        case "CHECK_HEALTH": {
          try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            const res = await fetch(`${baseUrl}/health`, { signal: controller.signal });
            clearTimeout(timeoutId);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            sendResponse({ success: true, data });
          } catch (err) {
            sendResponse({ success: false, error: err.message || "Failed to connect to PromptShield backend" });
          }
          break;
        }

        case "ANALYZE_PROMPT": {
          if (!config.enabled) {
            sendResponse({ success: true, bypassed: true });
            return;
          }

          try {
            const res = await fetch(`${baseUrl}/analyze`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                prompt: payload.prompt,
                security_level: config.securityLevel || "BALANCED",
              }),
            });

            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const data = await res.json();

            // Increment stats
            const stats = config.stats || DEFAULT_CONFIG.stats;
            stats.promptsScanned = (stats.promptsScanned || 0) + 1;
            if (data.detected_entities && data.detected_entities.length > 0) {
              stats.risksAverted = (stats.risksAverted || 0) + 1;
            }
            await chrome.storage.local.set({ stats });

            sendResponse({ success: true, data });
          } catch (err) {
            sendResponse({ success: false, error: err.message });
          }
          break;
        }

        case "SANITIZE_PROMPT": {
          try {
            const res = await fetch(`${baseUrl}/sanitize`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                prompt: payload.prompt,
                session_id: payload.session_id,
                security_level: config.securityLevel || "BALANCED",
                approved_entity_ids: payload.approved_entity_ids || [],
              }),
            });

            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const data = await res.json();

            // Update stats & persist last session ID for response restoration
            const stats = config.stats || DEFAULT_CONFIG.stats;
            stats.promptsMasked = (stats.promptsMasked || 0) + 1;
            stats.entitiesProtected = (stats.entitiesProtected || 0) + (data.masked_count || 0);
            await chrome.storage.local.set({ stats, lastSessionId: data.session_id });

            sendResponse({ success: true, data });
          } catch (err) {
            sendResponse({ success: false, error: err.message });
          }
          break;
        }

        case "RESTORE_RESPONSE": {
          try {
            const effectiveSessionId = payload.session_id || config.lastSessionId;
            if (!effectiveSessionId) {
              sendResponse({ success: false, error: "No active session ID for restoration" });
              return;
            }

            const res = await fetch(`${baseUrl}/restore`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                llm_response: payload.llm_response,
                session_id: effectiveSessionId,
                allow_all_restoration: payload.allow_all_restoration || false,
              }),
            });

            if (!res.ok) throw new Error(`HTTP error ${res.status}`);
            const data = await res.json();
            sendResponse({ success: true, data });
          } catch (err) {
            sendResponse({ success: false, error: err.message });
          }
          break;
        }

        default:
          sendResponse({ success: false, error: `Unknown message type: ${type}` });
      }
    } catch (fatalError) {
      sendResponse({ success: false, error: fatalError.message });
    }
  })();

  return true; // Keep message channel open for async response
});
