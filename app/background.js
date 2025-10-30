// ---------------------------
// Config
// ---------------------------
const BACKEND_BASE = "http://127.0.0.1:5000";
const TIMEOUT_MS = 120000;

// ---------------------------
// Per-tab Tracking 
// ---------------------------
const ongoingTextPromises = {};  
const lastTextPerTab = {};        
const textLocksPerTab = {};       
const ongoingImagePromises = {};  
const imageLocksPerTab = {};
const tabSessions = {};  

// ---------------------------
// Helper: Get or Create Session for Tab
// ---------------------------
function getSessionForTab(tabId) {
  if (!tabSessions[tabId]) {
    tabSessions[tabId] = crypto.randomUUID();
    console.log(`Created new session for tab ${tabId}:`, tabSessions[tabId]);
  }
  return tabSessions[tabId];
}

// Keep popup connection alive

// ---------------------------
// Helper: Clear Session for Tab
// ---------------------------
async function clearSessionForTab(tabId) {
  const sessionId = tabSessions[tabId];
  if (!sessionId) return;
  
  console.log(`Clearing session for tab ${tabId}:`, sessionId);
  
  try {
    await fetch(`${BACKEND_BASE}/cancel_session`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-ID": sessionId
      },
      body: JSON.stringify({ session_id: sessionId }),
      keepalive: true
    });
    
    console.log(`✅ Session cancelled for tab ${tabId}`);
  } catch (err) {
    console.error(`Error cancelling session for tab ${tabId}:`, err);
  } finally {
    // Clean up
    delete tabSessions[tabId];
    delete ongoingTextPromises[tabId];
    delete lastTextPerTab[tabId];
    delete textLocksPerTab[tabId];
    delete ongoingImagePromises[tabId];
    delete imageLocksPerTab[tabId];
  }
}

// ---------------------------
// Background Message Listener 
// ---------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message?.type) return;

  const tabId = sender.tab?.id;
  const sendToTab = (type, payload) => {
    if (tabId) {
      chrome.tabs.sendMessage(tabId, { type, payload }).catch(() => {
      });
    }
  };

  switch (message.type) {
    case "ANALYZE_TEXT_INITIAL": {
      const sessionId = message.payload?.session_id || getSessionForTab(tabId);

      fetch(`${BACKEND_BASE}/detect_text_initial`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId
        },
        body: JSON.stringify({
          text: message.payload?.text,
          url: message.payload?.url,
          session_id: sessionId
        })
      })
        .then(res => res.json())
        .then(data => {
          chrome.tabs.sendMessage(tabId, {
            type: "TEXT_INITIAL_RESULT",
            payload: {
              initial_analysis: data.initial_analysis,
              session_id: sessionId
            }
          });
        })
        .catch(err => {
          console.error("Initial analysis request failed:", err);
        });

      sendResponse({ received: true });
      return true;
    }

    case "ANALYZE_TEXT": {
      const sessionId = message.payload?.session_id || getSessionForTab(tabId);
      
      analyzeText(tabId, message.payload, sessionId)
        .then(result => {
          console.log("Text analysis result (raw):", result);
          const payload = {
            score: result.score || result.summary?.score || 0,
            explanation: result.explanation || result.summary?.explanation || "No explanation available.",
            prediction: result.prediction || result.summary?.prediction || "Unknown",
            input_text: result.input_text || message.payload?.text || "",
            session_id: sessionId,  
            article_id: result.article_id,
            source: result.source
          };
          console.log("✅ Processed text result:", payload);
          sendResponse(payload);
        })
        .catch(err => {
          console.error("Text analysis error:", err);
          const errorMsg = { 
            error: err.message || "Unknown error during text analysis.",
            session_id: sessionId
          };
          sendResponse(errorMsg);
          sendToTab("ANALYSIS_ERROR", errorMsg);
        });
      return true;
    }

    // ---------------------------
    // Image Analysis 
    // ---------------------------
    case "ANALYZE_IMAGE": {
      const sessionId = message.payload?.session_id || getSessionForTab(tabId);
      
      if (!tabId) {
        console.error("❌ No tab ID available for image analysis");
        sendResponse({ error: "No tab context available" });
        return false;
      }
      
      console.log(`🖼️ Processing image analysis for tab ${tabId}, session ${sessionId}`);
      
      analyzeImage(tabId, message.payload, sessionId)
        .then(results => {
          console.log(`✅ Image analysis complete for tab ${tabId}:`, results);
          
          const resultsWithSession = results.map(r => ({
            ...r,
            session_id: sessionId
          }));
          
          // Send each result to the tab's content script
          resultsWithSession.forEach(res => {
            console.log(`📤 Sending IMAGE_ANALYSIS_RESULT to tab ${tabId}:`, res);
            sendToTab("IMAGE_ANALYSIS_RESULT", res);
          });
          
          // sendResponse with count (older UI expectation)
          sendResponse({ success: true, count: results.length });
        })
        .catch(err => {
          console.error(`❌ Image analysis error for tab ${tabId}:`, err);
          const errorMsg = { 
            error: err.message || "Unknown error during image analysis.",
            session_id: sessionId
          };
          sendToTab("ANALYSIS_ERROR", errorMsg);
          sendResponse(errorMsg);
        });
      return true;
    }

    case "CANCEL_SESSION": {
      const sessionId = message.payload?.session_id;
      const targetTabId = message.payload?.tab_id || tabId;
      
      if (sessionId && tabSessions[targetTabId] === sessionId) {
        clearSessionForTab(targetTabId)
          .then(() => {
            sendResponse({ 
              status: "success", 
              message: "Session cancelled",
              session_id: sessionId,
              tab_id: targetTabId
            });
          })
          .catch(err => {
            sendResponse({ 
              status: "error", 
              error: err.message,
              session_id: sessionId
            });
          });
      } else {
        sendResponse({ 
          status: "error", 
          error: "Session not found or mismatch" 
        });
      }
      return true;
    }

    case "CHECK_SESSION_TASKS": {
      const sessionId = message.payload?.session_id || getSessionForTab(tabId);
      
      fetch(`${BACKEND_BASE}/session_tasks?session_id=${sessionId}`, {
        method: "GET",
        headers: {
          "X-Session-ID": sessionId
        }
      })
        .then(res => res.json())
        .then(data => {
          console.log("Active tasks:", data);
          sendResponse(data);
        })
        .catch(err => {
          console.error("Error checking tasks:", err);
          sendResponse({ error: err.message });
        });
      
      return true;
    }

    default:
      if (!message.type.endsWith("_RESULT")) {
        sendResponse({ error: "Unknown message type" });
      }
      return false;
  }
});

async function analyzeText(tabId, payload, sessionId) {
  const { text, url } = payload || {};
  if (!text?.trim()) throw new Error("No text provided.");

  // Ensure objects exist
  textLocksPerTab[tabId] = textLocksPerTab[tabId] || false;
  lastTextPerTab[tabId] = lastTextPerTab[tabId] || "";

  // **Older UI/flow behavior**: block new requests while lock is active and return a lock-response object
  if (textLocksPerTab[tabId]) {
    console.warn(`[Tab ${tabId}] Blocked new text request: analysis already in progress.`);
    return {
      success: false,
      source: "lock",
      input_text: text,
      session_id: sessionId,
      score: 0,
      explanation: "Another analysis is already running. Please wait.",
      prediction: "Unknown"
    };
  }

  if (ongoingTextPromises[tabId] && lastTextPerTab[tabId] === text) {
    console.warn(`[Tab ${tabId}] Duplicate text detected — returning previous promise`);
    return ongoingTextPromises[tabId];
  }

  lastTextPerTab[tabId] = text;
  textLocksPerTab[tabId] = true;

  const fetchOnce = async (url, options) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      return data;
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  };

  // Store the promise so duplicate requests reuse it
  ongoingTextPromises[tabId] = (async () => {
    try {
      const data = await fetchOnce(`${BACKEND_BASE}/detect_text`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Session-ID": sessionId
        },
        body: JSON.stringify({ text, url, session_id: sessionId })
      });
      return data;
    } finally {
      textLocksPerTab[tabId] = false;
      ongoingTextPromises[tabId] = null;
    }
  })();

  return ongoingTextPromises[tabId];
}

async function analyzeImage(tabId, payload, sessionId) {
  let urls = [];
  if (payload?.url) urls = [payload.url];
  else if (Array.isArray(payload?.urls)) urls = payload.urls;
  if (urls.length === 0) throw new Error("No image URLs provided.");

  imageLocksPerTab[tabId] = imageLocksPerTab[tabId] || false;

  if (imageLocksPerTab[tabId]) {
    console.warn(`[Tab ${tabId}] Blocked new image request: analysis already in progress.`);
    return urls.map(u => ({
      url: u,
      score: 0,
      explanation: "Another image analysis is in progress. Please wait.",
      prediction: "Unknown",
      session_id: sessionId
    }));
  }

  imageLocksPerTab[tabId] = true;

  const fetchOnce = async (url, options) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(url, { ...options, signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      return data;
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  };

  try {
    const data = await fetchOnce(`${BACKEND_BASE}/detect_image`, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "X-Session-ID": sessionId 
      },
      body: JSON.stringify({ 
        urls,
        session_id: sessionId 
      })
    });

    let resultsToSend = [];

    if (Array.isArray(data)) {
      resultsToSend = data.map((d, idx) => ({
        url: urls[idx],
        score: d.score || 0,
        explanation: d.explanation || "No explanation available.",
        prediction: d.prediction || "Unknown",
        cached: d.cached || false
      }));
    } else if (data.details && Array.isArray(data.details)) {
      resultsToSend = data.details.map((d, idx) => ({
        url: d.url || d.image_source || urls[idx],
        score: d.score || data.score || 0,
        explanation: d.explanation || data.explanation || "No explanation available.",
        prediction: d.verdict || d.prediction || "Unknown",
        cached: d.cached || false
      }));
    } else {
      resultsToSend = [{
        url: urls[0],
        score: data.score || 0,
        explanation: data.explanation || "No explanation available.",
        prediction: data.prediction || "Unknown",
        cached: data.cached || false
      }];
    }

    return resultsToSend;
  } finally {
    imageLocksPerTab[tabId] = false;
  }
}

chrome.tabs.onRemoved.addListener((tabId, removeInfo) => {
  console.log(`Tab ${tabId} closed, cancelling session...`);
  clearSessionForTab(tabId);
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'loading' && changeInfo.url) {
    console.log(`Tab ${tabId} navigating to ${changeInfo.url}, cancelling old session...`);
    clearSessionForTab(tabId);
  }
});

chrome.tabs.onReplaced?.addListener((addedTabId, removedTabId) => {
  console.log(`Tab ${removedTabId} replaced by ${addedTabId}, transferring session...`);
  if (tabSessions[removedTabId]) {
    tabSessions[addedTabId] = tabSessions[removedTabId];
    delete tabSessions[removedTabId];
  }
});

if (chrome.runtime.onSuspend) {
  chrome.runtime.onSuspend.addListener(() => {
    console.log("Extension suspending, cancelling all sessions...");
    Object.keys(tabSessions).forEach(tabId => {
      clearSessionForTab(parseInt(tabId));
    });
  });
}

setInterval(() => {
  console.log("Running periodic session cleanup...");
  
  chrome.tabs.query({}, (tabs) => {
    const activeTabIds = new Set(tabs.map(t => t.id));

    Object.keys(tabSessions).forEach(tabId => {
      const numericTabId = parseInt(tabId);
      if (!activeTabIds.has(numericTabId)) {
        console.log(`Cleaning up orphaned session for tab ${tabId}`);
        clearSessionForTab(numericTabId);
      }
    });
  });
}, 5 * 60 * 1000);
