// ================================
// content.js (Unified Handling)
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
badge.style.bottom = "16px";
badge.style.right = "16px";
badge.style.zIndex = 2147483647;
badge.style.padding = "8px 12px";
badge.style.borderRadius = "20px";
badge.style.background = "rgba(255,255,255,0.95)";
badge.style.boxShadow = "0 4px 12px rgba(0,0,0,0.15)";
badge.style.fontFamily = "Segoe UI, Roboto, Arial, sans-serif";
badge.style.fontSize = "14px";
badge.style.fontWeight = "600";
badge.style.cursor = "pointer";
badge.style.color = "#444";
badge.textContent = "Trust Score: —";
document.documentElement.appendChild(badge);

// ---------------------------
// PANEL (expanded)
// ---------------------------
const panel = document.createElement("div");
panel.id = "trustmeter-panel";
panel.style.position = "fixed";
panel.style.bottom = "16px";
panel.style.right = "16px";
panel.style.zIndex = 2147483647;
panel.style.minWidth = "280px";
panel.style.maxWidth = "420px";
panel.style.background = "rgba(255,255,255,0.97)";
panel.style.boxShadow = "0 6px 18px rgba(0,0,0,0.18)";
panel.style.borderRadius = "12px";
panel.style.fontFamily = "Segoe UI, Roboto, Arial, sans-serif";
panel.style.padding = "12px";
panel.style.fontSize = "13px";
panel.style.color = "#111";
panel.style.display = "none";
panel.style.flexDirection = "column";
panel.style.gap = "10px";

// Header
const header = document.createElement("div");
header.style.display = "flex";
header.style.justifyContent = "space-between";
const title = document.createElement("div");
title.textContent = "TrustMeter";
title.style.fontWeight = "600";
title.style.fontSize = "15px";
const closeBtn = document.createElement("button");
closeBtn.textContent = "×";
closeBtn.style.border = "none";
closeBtn.style.background = "transparent";
closeBtn.style.fontSize = "18px";
closeBtn.style.cursor = "pointer";
closeBtn.onclick = () => {
  panel.style.display = "none";
  badge.style.display = "inline-block";
};
header.appendChild(title);
header.appendChild(closeBtn);

// Score Row
const scoreRow = document.createElement("div");
scoreRow.style.display = "flex";
scoreRow.style.justifyContent = "space-between";
const scoreText = document.createElement("div");
scoreText.textContent = "Score: —";
scoreText.style.fontWeight = "700";
const refreshBtn = document.createElement("button");
refreshBtn.textContent = "Analyze";
refreshBtn.style.padding = "6px 10px";
refreshBtn.style.borderRadius = "6px";
refreshBtn.style.cursor = "pointer";
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
textSection.style.fontSize = "12px";
textSection.style.color = "#333";

const imageSection = document.createElement("div");
imageSection.innerHTML =
  "<strong>Image Analysis:</strong><br><div id='image-results'>No images analyzed.</div>";
imageSection.style.fontSize = "12px";
imageSection.style.color = "#333";

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
  badge.style.color = "#444";
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
  if (avg >= 75) badge.style.color = "#0b8043";
  else if (avg >= 45) badge.style.color = "#e09b00";
  else badge.style.color = "#c42f2f";
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
        imgEntry.innerHTML = `
          <img src="${imgUrl}" style="max-width:80px; max-height:50px; margin-right:6px; vertical-align:middle;"> 
          Validity: <strong>${validity}%</strong>
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
  panel.style.display = "flex";
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
      imgEntry.innerHTML = `
        <img src="${url}" style="max-width:80px; max-height:50px; margin-right:6px; vertical-align:middle;"> 
        Validity: <strong>${validity}%</strong>
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