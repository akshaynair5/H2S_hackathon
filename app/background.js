// ---------------------------
// background.js (Improved)
// ---------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message?.type) return;

  const tabId = sender.tab?.id;

  switch (message.type) {
    case "ANALYZE_TEXT":
      analyzeText(message.payload)
        .then(result => {
          sendResponse(result);
          if (tabId) {
            chrome.tabs.sendMessage(tabId, {
              type: "TEXT_ANALYSIS_RESULT",
              payload: result
            });
          }
        })
        .catch(err => {
          const errorMsg = { error: err.message };
          sendResponse(errorMsg);
          if (tabId) {
            chrome.tabs.sendMessage(tabId, {
              type: "ANALYSIS_ERROR",
              payload: err.message
            });
          }
        });
      return true;

    case "ANALYZE_IMAGE":
      analyzeImage(message.payload)
        .then(results => {
          sendResponse(results);
          if (tabId) {
            results.forEach(res => {
              chrome.tabs.sendMessage(tabId, {
                type: "IMAGE_ANALYSIS_RESULT",
                payload: res
              });
            });
          }
        })
        .catch(err => {
          const errorMsg = { error: err.message };
          sendResponse(errorMsg);
          if (tabId) {
            chrome.tabs.sendMessage(tabId, {
              type: "ANALYSIS_ERROR",
              payload: err.message
            });
          }
        });
      return true;

    default:
      if (!message.type.endsWith("_RESULT")) {
        sendResponse({ error: "Unknown message type" });
      }
      return false;
  }
});

// ---------------------------
// Text Analysis
// ---------------------------
let lastText = "";
async function analyzeText(payload) {
  const { text } = payload || {};
  if (!text) throw new Error("No text provided");

  if (text === lastText) {
    return { score: 0, explanation: "Duplicate request ignored (cached)." };
  }
  lastText = text;

  const res = await fetch("http://127.0.0.1:5000/detect_text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text })
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ---------------------------
// Image Analysis
// ---------------------------
let lastImageUrls = new Set();
async function analyzeImage(payload) {
  let urls = [];
  if (payload?.url) urls = [payload.url];
  else if (Array.isArray(payload?.urls)) urls = payload.urls;
  if (urls.length === 0) throw new Error("No image URLs provided");

  const newUrls = urls.filter(u => !lastImageUrls.has(u));
  if (newUrls.length === 0) {
    return [{ score: 0, explanation: "Duplicate request ignored (cached)." }];
  }
  newUrls.forEach(u => lastImageUrls.add(u));

  const res = await fetch("http://127.0.0.1:5000/detect_image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls: newUrls })
  });

  const data = await res.json();
  if (data.error) throw new Error(data.error);

  if (Array.isArray(data)) {
    return data.map((d, idx) => ({
      url: newUrls[idx],
      score: d.score,
      explanation: d.explanation || "No explanation available."
    }));
  } else {
    return [
      {
        url: newUrls[0],
        score: data.score,
        explanation: data.explanation || "No explanation available."
      }
    ];
  }
}