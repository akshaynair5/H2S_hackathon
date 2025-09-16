// ---------------------------
// background.js (Unified forwarding)
// ---------------------------

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message.type) return;

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
          sendResponse({ error: err.message });
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
          sendResponse({ error: err.message });
          if (tabId) {
            chrome.tabs.sendMessage(tabId, {
              type: "ANALYSIS_ERROR",
              payload: err.message
            });
          }
        });
      return true; 

    default:
      sendResponse({ error: "Unknown message type" });
      return false;
  }
});

// ---------------------------
// Helper functions
// ---------------------------

async function analyzeText(payload) {
  const { text } = payload;
  if (!text) throw new Error("No text provided");

  try {
    const res = await fetch("http://127.0.0.1:5000/detect_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    return data; 
  } catch (err) {
    throw new Error("Text analysis failed: " + err.message);
  }
}

async function analyzeImage(payload) {
  let urls = [];
  if (payload.url) urls = [payload.url];
  else if (Array.isArray(payload.urls)) urls = payload.urls;

  if (urls.length === 0) throw new Error("No image URLs provided");

  try {
    const res = await fetch("http://127.0.0.1:5000/detect_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls })
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (Array.isArray(data)) {
      return data.map((d, idx) => ({
        url: urls[idx],
        score: d.score,
        explanation: d.explanation || "No explanation available."
      }));
    } else {
      return [
        {
          url: urls[0],
          score: data.score,
          explanation: data.explanation || "No explanation available."
        }
      ];
    }
  } catch (err) {
    throw new Error("Image analysis failed: " + err.message);
  }
}
