// ---------------------------
// popup.js
// ---------------------------

// // Open Options page
// document.getElementById("openOptions").addEventListener("click", () => {
//   try { 
//     chrome.runtime.openOptionsPage(); 
//   } catch (e) { 
//     alert("Open options failed"); 
//   }
// });

chrome.storage.local.get({ lastResult: null }, (items) => {
  if (items.lastResult) {
    displayResult(items.lastResult);
  }
});

// ---------------------------
// TEXT CHECK
// ---------------------------
document.getElementById("checkText").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString()
  }, async (results) => {
    const text = results[0].result || "";
    if (!text.trim()) {
      alert("Please select some text on the page first!");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/detect_text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      if (data.error) {
        alert("Error: " + data.error);
      } else {
        displayResult(data);
      }
    } catch (err) {
      alert("Request failed: " + err);
    }
  });
});

// ---------------------------
// IMAGE CHECK (ALL)
// ---------------------------
document.getElementById("checkAllImages").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => Array.from(document.querySelectorAll("img"))
                    .map(img => img.src)
                    .filter(src => src.startsWith("http"))
  }, async (results) => {
    const imageUrls = results[0].result;
    if (!imageUrls.length) {
      displayError("⚠️ No images found on this page.");
      return;
    }

    try {
      const response = await fetch("http://127.0.0.1:5000/detect_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: imageUrls })
      });
      const data = await response.json();

      displayResult({
        score: data[0]?.score || 0,
        explanation: "Multiple images analyzed",
        details: Array.isArray(data) ? data : [data]
      });
    } catch (err) {
      displayError("Error: " + err.message);
    }
  });
});

// ---------------------------
// IMAGE CHECK (CLICK-TO-CHECK)
// ---------------------------
document.getElementById("clickToCheck").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      document.querySelectorAll(".misinfo-overlay").forEach(el => el.remove());
      document.querySelectorAll("img").forEach(img => {
        const overlay = document.createElement("div");
        overlay.innerText = "Check Image";
        overlay.style.position = "absolute";
        overlay.style.background = "rgba(0,0,0,0.7)";
        overlay.style.color = "white";
        overlay.style.padding = "2px 4px";
        overlay.style.fontSize = "12px";
        overlay.style.cursor = "pointer";
        overlay.style.zIndex = "9999";
        const rect = img.getBoundingClientRect();
        overlay.style.top = `${window.scrollY + rect.top}px`;
        overlay.style.left = `${window.scrollX + rect.left}px`;
        overlay.className = "misinfo-overlay";
        document.body.appendChild(overlay);

        overlay.addEventListener("click", () => {
          chrome.runtime.sendMessage({ action: "checkImage", src: img.src });
        });
      });
    }
  });
});

// ---------------------------
// MESSAGE LISTENER FOR CLICK-TO-CHECK
// ---------------------------
chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "checkImage") {
    fetch("http://127.0.0.1:5000/detect_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls: [message.src] })
    })
      .then(res => res.json())
      .then(data => {
        const result = Array.isArray(data) ? data[0] : data;
        displayResult(result);
      })
      .catch(err => displayError("Request failed: " + err));
  }
});

// ---------------------------
// DISPLAY RESULT FUNCTION
// ---------------------------
function displayResult(data) {
  const lastDiv = document.getElementById("last");
  const explainDiv = document.getElementById("explain");

  // Overall score
  lastDiv.textContent = Math.round((data.score || 0) * 100) + "%";
  lastDiv.style.color = (data.score >= 0.75) ? "#0b8043" : (data.score >= 0.45) ? "#e09b00" : "#c42f2f";

  explainDiv.innerHTML = "";

  if (data.details && Array.isArray(data.details)) {
    data.details.forEach((imgRes, idx) => {
      const div = document.createElement("div");
      div.style.marginBottom = "6px";
      div.style.fontSize = "12px";
      div.innerHTML = `<b>Image ${idx + 1}:</b> ${Math.round(imgRes.score * 100)}% valid — ${imgRes.explanation}`;
      explainDiv.appendChild(div);
    });
  } else {
    explainDiv.textContent = data.explanation || "No explanation returned.";
  }
  chrome.storage.local.set({ lastResult: data });
}

// ---------------------------
// DISPLAY ERROR FUNCTION
// ---------------------------
function displayError(msg) {
  const lastDiv = document.getElementById("last");
  const explainDiv = document.getElementById("explain");

  lastDiv.textContent = "—";
  lastDiv.style.color = "#c42f2f";
  explainDiv.textContent = msg;
}
