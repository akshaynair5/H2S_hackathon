// background.js
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "ANALYZE_TEXT") {
    (async () => {
      try {

        const opts = await chrome.storage.sync.get({ backendUrl: "", optIn: true });
        if (!opts.optIn) {
          sendResponse({ error: "User opted out of sending page content." });
          return;
        }
        const backendUrl = opts.backendUrl || "http://localhost:8000/score";
  
        const payload = {
          text: msg.payload.text,
          url: msg.payload.url,
  
          userAgent: navigator.userAgent,
          timestamp: new Date().toISOString()
        };
        const res = await fetch(backendUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const text = await res.text();
          sendResponse({ error: `Backend error ${res.status}: ${text}` });
          return;
        }
        const data = await res.json();
        sendResponse(data);
      } catch (err) {
        sendResponse({ error: err.message || String(err) });
      }
    })();

    return true;
  }
});
