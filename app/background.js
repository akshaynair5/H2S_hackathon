// ---------------------------
// Background Message Listener
// ---------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message?.type) return;

  const tabId = sender.tab?.id;
  const sendToTab = (type, payload) => {
    if (tabId) chrome.tabs.sendMessage(tabId, { type, payload });
  };

  switch (message.type) {
    // ---------------------------
    // Text Analysis
    // ---------------------------
    case "ANALYZE_TEXT":
      analyzeText(tabId, message.payload)
        .then(result => {
          console.log("Text analysis result (raw):", result);
          const payload = {
            score: result.score || result.summary?.score || 0,
            explanation: result.explanation || result.summary?.explanation || "No explanation available.",
            prediction: result.prediction || result.summary?.prediction || "Unknown",
            input_text: result.input_text || message.payload?.text || ""
          };
          console.log("✅ Processed text result:", payload);
          sendResponse(payload);
        })
        .catch(err => {
          console.error("Text analysis error:", err);
          const errorMsg = { error: err.message || "Unknown error during text analysis." };
          sendResponse(errorMsg);
          sendToTab("ANALYSIS_ERROR", errorMsg);
        });
      return true;
    // ---------------------------
    // Image Analysis
    // ---------------------------
    case "ANALYZE_IMAGE":
      analyzeImage(tabId, message.payload)
        .then(results => {
          sendResponse(results);
          results.forEach(res => sendToTab("IMAGE_ANALYSIS_RESULT", res));
        })
        .catch(err => {
          console.error("Image analysis error:", err);
          const errorMsg = { error: err.message || "Unknown error during image analysis." };
          sendResponse(errorMsg);
          sendToTab("ANALYSIS_ERROR", errorMsg);
        });
      return true;

    default:
      if (!message.type.endsWith("_RESULT")) sendResponse({ error: "Unknown message type" });
      return false;
  }
});

// ---------------------------
// Config
// ---------------------------
const BACKEND_BASE = "http://127.0.0.1:5000";
const TIMEOUT_MS = 120000;
// const RETRY_LIMIT = 2; // Disabled retries for now

// ---------------------------
// Per-tab Tracking
// ---------------------------
const ongoingTextPromises = {};  
const lastTextPerTab = {};        
const textLocksPerTab = {};       
const ongoingImagePromises = {};  
const imageLocksPerTab = {};     

async function analyzeText(tabId, payload) {
  const { text } = payload || {};
  if (!text?.trim()) throw new Error("No text provided.");

  textLocksPerTab[tabId] = textLocksPerTab[tabId] || false;
  lastTextPerTab[tabId] = lastTextPerTab[tabId] || "";

  if (textLocksPerTab[tabId]) {
    console.warn(`[Tab ${tabId}] Blocked new text request: analysis already in progress.`);
    return {
      success: false,
      source: "lock",
      input_text: text,
      result: {
        overall: { label: "Unknown", confidence: 0 },
        explanation: "Another analysis is already running. Please wait."
      }
    };
  }

  if (ongoingTextPromises[tabId] && lastTextPerTab[tabId] === text) {
    console.warn(`[Tab ${tabId}] Duplicate text request detected; using existing promise.`);
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
      throw err; // No automatic retry
    }
  };

  ongoingTextPromises[tabId] = (async () => {
    try {
      const data = await fetchOnce(`${BACKEND_BASE}/detect_text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      return data;
    } finally {
      textLocksPerTab[tabId] = false;
      ongoingTextPromises[tabId] = null;
    }
  })();

  return ongoingTextPromises[tabId];
}

// ---------------------------
// IMAGE ANALYSIS
// ---------------------------
async function analyzeImage(tabId, payload) {
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
      prediction: "Unknown"
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
      throw err; // No automatic retry
    }
  };

  try {
    const data = await fetchOnce(`${BACKEND_BASE}/detect_image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls })
    });

    let resultsToSend = [];

    // To handle both single-image and multi-image structures
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
        url: d.url || urls[idx],
        score: d.score || data.score || 0,
        explanation: d.explanation || data.explanation || "No explanation available.",
        prediction: d.prediction || "Unknown",
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
