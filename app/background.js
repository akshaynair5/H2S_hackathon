
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message.type) return;

  switch (message.type) {
    case "ANALYZE_TEXT":
      analyzeText(message.payload)
        .then(result => sendResponse(result))
        .catch(err => sendResponse({ error: err.message }));
      return true;

    case "ANALYZE_IMAGE":
      analyzeImage(message.payload)
        .then(result => sendResponse(result))
        .catch(err => sendResponse({ error: err.message }));
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
  const { url } = payload;
  if (!url) throw new Error("No image URL provided");

  try {
    const res = await fetch("http://127.0.0.1:5000/detect_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls: [url] })
    });
    const data = await res.json();
    return Array.isArray(data) ? data[0] : data;
  } catch (err) {
    throw new Error("Image analysis failed: " + err.message);
  }
}
