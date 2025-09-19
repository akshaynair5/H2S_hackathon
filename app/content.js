// ================================
// content.js (Unified Handling with Improved Styles)
// ================================

// ---------------------------
// NORMALIZE SCORE
// ---------------------------
function normalizeScore(raw) {
  let s = raw || 0;
  if (s <= 1) s = Math.round(s * 100);
  else s = Math.round(s);
  return Math.max(0, Math.min(100, s));
}

// ---------------------------
// BADGE (collapsed)
// ---------------------------
const badge = document.createElement("div");
badge.id = "trustmeter-badge";
badge.style.position = "fixed";
badge.style.bottom = "24px";
badge.style.right = "24px";
badge.style.zIndex = "2147483647";
badge.style.padding = "10px 16px";
badge.style.borderRadius = "12px";
badge.style.background = "linear-gradient(135deg, #667eea, #764ba2)"; // Match main.css gradient
badge.style.color = "#ffffff";
badge.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
badge.style.fontSize = "14px";
badge.style.fontWeight = "600";
badge.style.cursor = "pointer";
badge.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.2)";
badge.style.transition = "transform 0.2s ease, box-shadow 0.2s ease";
badge.style.display = "flex";
badge.style.alignItems = "center";
badge.style.gap = "8px";
badge.textContent = "Trust Score: —";
badge.setAttribute("role", "button"); 
badge.setAttribute("aria-label", "Open TrustMeter panel");
badge.addEventListener("mouseover", () => {
  badge.style.transform = "translateY(-2px)";
  badge.style.boxShadow = "0 6px 16px rgba(0, 0, 0, 0.25)";
});
badge.addEventListener("mouseout", () => {
  badge.style.transform = "translateY(0)";
  badge.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.2)";
});
document.documentElement.appendChild(badge);

// ---------------------------
// PANEL (expanded)
// ---------------------------
const panel = document.createElement("div");
panel.id = "trustmeter-panel";
panel.style.position = "fixed";
panel.style.bottom = "24px";
panel.style.right = "24px";
panel.style.zIndex = "2147483647";
panel.style.minWidth = "300px";
panel.style.maxWidth = "400px";
panel.style.background = "rgba(255, 255, 255, 0.95)";
panel.style.backdropFilter = "blur(10px)"; 
panel.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.2)";
panel.style.borderRadius = "16px";
panel.style.border = "1px solid rgba(255, 255, 255, 0.2)";
panel.style.fontFamily = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
panel.style.padding = "16px";
panel.style.color = "#2d3748";
panel.style.display = "none";
panel.style.flexDirection = "column";
panel.style.gap = "12px";
panel.style.maxHeight = "80vh";
panel.style.overflowY = "auto";
panel.style.transition = "opacity 0.3s ease, transform 0.3s ease";
panel.style.opacity = "0";
panel.style.transform = "translateY(10px)";
panel.setAttribute("role", "dialog"); 
panel.setAttribute("aria-label", "TrustMeter Analysis Panel");

// Show panel animation
function showPanel() {
  panel.style.display = "flex";
  setTimeout(() => {
    panel.style.opacity = "1";
    panel.style.transform = "translateY(0)";
  }, 10);
}

// Header
const header = document.createElement("div");
header.style.display = "flex";
header.style.alignItems = "center";
header.style.justifyContent = "space-between";
header.style.paddingBottom = "8px";
header.style.borderBottom = "1px solid rgba(0, 0, 0, 0.1)";
const title = document.createElement("div");
title.textContent = "TrustMeter";
title.style.fontWeight = "700";
title.style.fontSize = "16px";
title.style.background = "linear-gradient(135deg, #667eea, #764ba2)"; // Match main.css gradient
title.style.webkitBackgroundClip = "text";
title.style.webkitTextFillColor = "transparent";
const closeBtn = document.createElement("button");
closeBtn.textContent = "×";
closeBtn.style.border = "none";
closeBtn.style.background = "transparent";
closeBtn.style.fontSize = "18px";
closeBtn.style.fontWeight = "600";
closeBtn.style.color = "#6b7280";
closeBtn.style.cursor = "pointer";
closeBtn.style.transition = "color 0.2s ease";
closeBtn.setAttribute("aria-label", "Close TrustMeter panel");
closeBtn.addEventListener("mouseover", () => {
  closeBtn.style.color = "#3b82f6";
});
closeBtn.addEventListener("mouseout", () => {
  closeBtn.style.color = "#6b7280";
});
closeBtn.onclick = () => {
  panel.style.opacity = "0";
  panel.style.transform = "translateY(10px)";
  setTimeout(() => {
    panel.style.display = "none";
    badge.style.display = "flex";
  }, 300);
};
header.appendChild(title);
header.appendChild(closeBtn);

// Score Row
const scoreRow = document.createElement("div");
scoreRow.style.display = "flex";
scoreRow.style.alignItems = "center";
scoreRow.style.justifyContent = "space-between";
scoreRow.style.padding = "8px 0";
const scoreText = document.createElement("div");
scoreText.textContent = "Score: —";
scoreText.style.fontWeight = "700";
scoreText.style.fontSize = "14px";
const refreshBtn = document.createElement("button");
refreshBtn.textContent = "Analyze";
refreshBtn.style.padding = "6px 12px";
refreshBtn.style.borderRadius = "8px";
refreshBtn.style.border = "none";
refreshBtn.style.background = "linear-gradient(135deg, #667eea, #764ba2)"; // Match main.css button gradient
refreshBtn.style.color = "#ffffff";
refreshBtn.style.fontSize = "13px";
refreshBtn.style.fontWeight = "600";
refreshBtn.style.cursor = "pointer";
refreshBtn.style.transition = "transform 0.2s ease, box-shadow 0.2s ease";
refreshBtn.setAttribute("aria-label", "Re-analyze content");
refreshBtn.addEventListener("mouseover", () => {
  refreshBtn.style.transform = "translateY(-1px)";
  refreshBtn.style.boxShadow = "0 4px 12px rgba(102, 126, 234, 0.3)";
});
refreshBtn.addEventListener("mouseout", () => {
  refreshBtn.style.transform = "translateY(0)";
  refreshBtn.style.boxShadow = "none";
});
refreshBtn.onclick = () => {
  analyzeTextNow();
  analyzeImagesNow();
};
scoreRow.appendChild(scoreText);
scoreRow.appendChild(refreshBtn);

// Sections
const textSection = document.createElement("div");
textSection.innerHTML =
  "<strong>Text Analysis:</strong><br><span id='text-result'>No analysis yet.</span>";
textSection.style.fontSize = "13px";
textSection.style.color = "#2d3748";
textSection.style.lineHeight = "1.5";
textSection.style.padding = "8px 0";

const imageSection = document.createElement("div");
imageSection.innerHTML =
  "<strong>Image Analysis:</strong><br><div id='image-results'>No images analyzed.</div>";
imageSection.style.fontSize = "13px";
imageSection.style.color = "#2d3748";
imageSection.style.lineHeight = "1.5";
imageSection.style.padding = "8px 0";

panel.appendChild(header);
panel.appendChild(scoreRow);
panel.appendChild(textSection);
panel.appendChild(imageSection);
document.documentElement.appendChild(panel);

// ---------------------------
// STATE
// ---------------------------
let collectedScores = [];

// ---------------------------
// HELPERS (UI)
// ---------------------------
function setWorking(msg) {
  scoreText.textContent = "Score: …";
  document.getElementById("text-result").textContent = msg || "Analyzing...";
}

function setError(err) {
  scoreText.textContent = "Score: —";
  document.getElementById("text-result").textContent = "Error: " + err;
  badge.textContent = "Trust Score: —";
  badge.style.color = "#ffffff"; // Use white for consistency with gradient
}

function setResult(score, explanation) {
  const percent = normalizeScore(score);
  scoreText.textContent = `Score: ${percent}%`;
  if (percent >= 75) scoreText.style.color = "#0b8043";
  else if (percent >= 45) scoreText.style.color = "#e09b00";
  else scoreText.style.color = "#c42f2f";

  document.getElementById("text-result").textContent =
    explanation || "No explanation returned.";

  collectedScores.push(percent);
  updateBadge();
}

function updateBadge() {
  if (collectedScores.length === 0) return;
  const avg = Math.round(
    collectedScores.reduce((a, b) => a + b, 0) / collectedScores.length
  );
  badge.textContent = `Trust Score: ${avg}%`;
  if (avg >= 75) badge.style.color = "#ffffff"; // White text for contrast on gradient
  else if (avg >= 45) badge.style.color = "#ffffff";
  else badge.style.color = "#ffffff";
}

// ---------------------------
// HELPERS (text + images)
// ---------------------------
function collectVisibleText(maxChars = 20000) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
      const style = window.getComputedStyle(node.parentElement);
      if (style && style.display === "none") return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  let text = "";
  let node;
  while ((node = walker.nextNode()) && text.length < maxChars) {
    text += node.nodeValue + " ";
  }
  return text.trim().substring(0, maxChars);
}

function collectVisibleImages(maxCount = 3) {
  return Array.from(document.images)
    .filter(img => img.src.startsWith("http") && img.offsetWidth > 10 && img.offsetHeight > 10)
    .slice(0, maxCount)
    .map(img => img.src);
}

// ---------------------------
// ANALYZE FUNCTIONS
// ---------------------------
function analyzeTextNow() {
  setWorking("Analyzing visible text...");
  collectedScores = [];

  const visibleText = collectVisibleText(45000);
  if (!visibleText) {
    setError("No visible text found.");
    return;
  }

  chrome.runtime.sendMessage(
    { type: "ANALYZE_TEXT", payload: { text: visibleText, url: location.href } },
    (response) => {
      if (!response || response.error) {
        setError(response?.error || "No response from backend.");
        return;
      }
      setResult(response.score ?? 0, response.explanation);
    }
  );
}

function analyzeImagesNow() {
  const container = document.getElementById("image-results");
  container.innerHTML = "";

  const visibleImages = collectVisibleImages(3);
  if (visibleImages.length === 0) {
    container.textContent = "No images found.";
    return;
  }

  chrome.runtime.sendMessage(
    { type: "ANALYZE_IMAGE", payload: { urls: visibleImages } },
    (response) => {
      if (!response || response.error) {
        container.textContent = response?.error || "Image analysis failed.";
        return;
      }

      response.forEach((imgResult, idx) => {
        const imgUrl = visibleImages[idx];
        const validity = normalizeScore(imgResult.score);

        const imgEntry = document.createElement("div");
        imgEntry.style.display = "flex";
        imgEntry.style.alignItems = "center";
        imgEntry.style.padding = "8px 0";
        imgEntry.style.borderTop = idx > 0 ? "1px solid rgba(0, 0, 0, 0.1)" : "none";
        imgEntry.innerHTML = `
          <img src="${imgUrl}" style="max-width: 80px; max-height: 50px; margin-right: 12px; border-radius: 4px; object-fit: cover;">
          <span>Validity: <strong>${validity}%</strong></span>
        `;
        if (validity >= 75) imgEntry.style.color = "#0b8043";
        else if (validity >= 45) imgEntry.style.color = "#e09b00";
        else imgEntry.style.color = "#c42f2f";

        container.appendChild(imgEntry);
        collectedScores.push(validity);
        updateBadge();
      });
    }
  );
}

// ---------------------------
// TOGGLE PANEL
// ---------------------------
badge.onclick = () => {
  badge.style.display = "none";
  showPanel();
};

// ---------------------------
// MESSAGE HANDLER (popup/bg → content)
// ---------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) return;

  switch (message.type) {
    case "TEXT_ANALYSIS_RESULT":
      setResult(message.payload.score, message.payload.explanation);
      break;

    case "IMAGE_ANALYSIS_RESULT": {
      const { url, score } = message.payload;
      const container = document.getElementById("image-results");
      const validity = normalizeScore(score);

      const imgEntry = document.createElement("div");
      imgEntry.style.display = "flex";
      imgEntry.style.alignItems = "center";
      imgEntry.style.padding = "8px 0";
      imgEntry.style.borderTop = container.children.length > 0 ? "1px solid rgba(0, 0, 0, 0.1)" : "none";
      imgEntry.innerHTML = `
        <img src="${url}" style="max-width: 80px; max-height: 50px; margin-right: 12px; border-radius: 4px; object-fit: cover;">
        <span>Validity: <strong>${validity}%</strong></span>
      `;
      if (validity >= 75) imgEntry.style.color = "#0b8043";
      else if (validity >= 45) imgEntry.style.color = "#e09b00";
      else imgEntry.style.color = "#c42f2f";

      container.appendChild(imgEntry);
      collectedScores.push(validity);
      updateBadge();
      break;
    }

    case "ANALYSIS_ERROR":
      setError(message.payload);
      break;
  }
});

// ---------------------------
// AUTO RUN (text only on load)
// ---------------------------
setTimeout(analyzeTextNow, 1000);