// ---------------------------
// UTIL: DISPLAY RESULT
// ---------------------------
function displayResult(result) {
  const resultsDiv = document.getElementById("results");

  const card = document.createElement("div");
  card.className =
    "result-card border p-3 mb-2 rounded bg-gray-50 shadow-sm text-sm";

  card.innerHTML = `
    <p><strong>Trust Score:</strong> ${result.score}%</p>
    <p><strong>Reasoning:</strong> ${result.explanation}</p>
    <details>
      <summary class="cursor-pointer">Details</summary>
      <pre class="whitespace-pre-wrap text-xs mt-1">${JSON.stringify(
        result.details,
        null,
        2
      )}</pre>
    </details>
  `;

  resultsDiv.appendChild(card);
}

function displayError(message) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = `<p class="text-red-600 font-semibold">${message}</p>`;
}

// ---------------------------
// ANALYZE TEXT
// ---------------------------
document.getElementById("checkText").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    {
      target: { tabId: tab.id },
      func: () => document.body.innerText,
    },
    (results) => {
      if (chrome.runtime.lastError || !results || !results[0]) {
        displayError("Failed to extract text from page.");
        return;
      }

      const textContent = results[0].result;
      chrome.runtime.sendMessage(
        { type: "ANALYZE_TEXT", payload: { text: textContent } },
        (response) => {
          if (!response || response.error) {
            displayError(response?.error || "Text analysis failed.");
            return;
          }
          displayResult({
            score: response.score || 0,
            explanation: response.explanation || "Text analyzed",
            details: response.details || [response],
          });
        }
      );
    }
  );
});

// ---------------------------
// ANALYZE ALL IMAGES
// ---------------------------
document.getElementById("clickToCheck").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    {
      target: { tabId: tab.id },
      func: () => Array.from(document.querySelectorAll("img"))
        .filter(img => img.offsetWidth > 10 && img.offsetHeight > 10)
        .map(img => img.src),
    },
    (results) => {
      if (!results || !results[0] || results[0].result.length === 0) {
        displayError("No images found.");
        return;
      }

      const imageUrls = results[0].result;
      chrome.runtime.sendMessage(
        { type: "ANALYZE_IMAGE", payload: { urls: imageUrls } },
        (response) => {
          if (!response || response.error) {
            displayError(response?.error || "Image analysis failed.");
            return;
          }

          response.forEach((imgRes) => {
            const wrapped = {
              score: imgRes.score || 0,
              explanation: imgRes.explanation || "Image analyzed.",
              details: [imgRes],
            };
            displayResult(wrapped);
          });
        }
      );
    }
  );
});

// ---------------------------
// CLICK-TO-CHECK IMAGES
// ---------------------------
document.getElementById("clickToCheck").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      function addOverlay(img) {
        const parent = img.parentElement;
        if (!parent) return;

        if (getComputedStyle(parent).position === "static") {
          parent.style.position = "relative";
        }

        const existing = parent.querySelector(".misinfo-overlay");
        if (existing) existing.remove();

        const overlay = document.createElement("div");
        overlay.innerText = "Check";
        overlay.style.position = "absolute";
        overlay.style.top = "5px";
        overlay.style.right = "5px";
        overlay.style.background = "rgba(0,0,0,0.7)";
        overlay.style.color = "white";
        overlay.style.padding = "2px 6px";
        overlay.style.fontSize = "11px";
        overlay.style.borderRadius = "4px";
        overlay.style.cursor = "pointer";
        overlay.style.zIndex = "9999";
        overlay.className = "misinfo-overlay";

        parent.appendChild(overlay);

        overlay.onclick = (e) => {
          e.stopPropagation();
          document.querySelectorAll(".misinfo-overlay").forEach(el => el.remove());

          chrome.runtime.sendMessage(
            { type: "ANALYZE_IMAGE", payload: { urls: [img.src] } }
          );
        };
      }

      const imgs = Array.from(document.querySelectorAll("img")).filter(
        img => img.offsetWidth > 10 && img.offsetHeight > 10
      );
      imgs.forEach(addOverlay);
    },
  });
});

// ---------------------------
// RECEIVE IMAGE ANALYSIS RESULTS AND UPDATE POPUP
// ---------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) return;

  if (message.type === "IMAGE_ANALYSIS_RESULT") {
    const { url, score, explanation } = message.payload;

    const wrapped = {
      score: score || 0,
      explanation: explanation || "Image analyzed",
      details: [message.payload],
    };
    displayResult(wrapped);
  }

  if (message.type === "ANALYSIS_ERROR") {
    displayError(message.payload);
  }
});
