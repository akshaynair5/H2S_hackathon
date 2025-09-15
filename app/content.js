function normalizeScore(raw) {
  let s = raw || 0;
  if (s <= 1) {
    s = Math.round(s * 100);
  } else {
    s = Math.round(s);
  }
  return Math.max(0, Math.min(100, s));
}

// ---------------------------
// COLLAPSED BADGE (DEFAULT VIEW)
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
// FULL PANEL (HIDDEN INITIALLY)
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
panel.style.display = "none"; // initially hidden
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
refreshBtn.onclick = analyzeNow;
scoreRow.appendChild(scoreText);
scoreRow.appendChild(refreshBtn);

const textSection = document.createElement("div");
textSection.innerHTML = "<strong>Text Analysis:</strong><br><span id='text-result'>No analysis yet.</span>";
textSection.style.fontSize = "12px";
textSection.style.color = "#333";

const imageSection = document.createElement("div");
imageSection.innerHTML = "<strong>Image Analysis:</strong><br><div id='image-results'>No images analyzed.</div>";
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
// UI STATE HELPERS
// ---------------------------
function setWorking(msg) {
  scoreText.textContent = "Score: …";
  document.getElementById("text-result").textContent = msg || "Analyzing...";
}

function setError(err) {
  scoreText.textContent = "Score: —";
  document.getElementById("text-result").textContent = "Error: " + err;
  badge.textContent = "Trust Score: —";
}

function setResult(score, shortExplanation) {
  let percent = normalizeScore(score);
  scoreText.textContent = `Score: ${percent}%`;
  if (score >= 0.75) scoreText.style.color = "#0b8043";
  else if (score >= 0.45) scoreText.style.color = "#e09b00";
  else scoreText.style.color = "#c42f2f";

  document.getElementById("text-result").textContent =
    shortExplanation || "No explanation returned.";

  collectedScores.push(percent);
  updateBadge();
}

function updateBadge() {
  if (collectedScores.length === 0) return;
  const avg = Math.round(collectedScores.reduce((a, b) => a + b, 0) / collectedScores.length);
  badge.textContent = `Trust Score: ${avg}%`;
  if (avg >= 75) badge.style.color = "#0b8043";
  else if (avg >= 45) badge.style.color = "#e09b00";
  else badge.style.color = "#c42f2f";
}

// ---------------------------
// ANALYZE FUNCTION
// ---------------------------
function analyzeNow() {
  setWorking("Analyzing visible text & images...");
  collectedScores = [];

  const visibleText = collectVisibleText(45000);
  const visibleImages = collectVisibleImages(3);

  if (!visibleText && visibleImages.length === 0) {
    setError("No text or images in viewport.");
    return;
  }

  if (visibleText) {
    chrome.runtime.sendMessage(
      { type: "ANALYZE_TEXT", payload: { text: visibleText, url: location.href } },
      (response) => {
        if (!response || response.error) {
          setError(response ? response.error : "No response from backend.");
          return;
        }
        setResult(response.score || 0, response.explanation || "");
      }
    );
  }

  const imageResultsContainer = document.getElementById("image-results");
  imageResultsContainer.innerHTML = "";
  visibleImages.forEach((imgUrl) => {
    const imgEntry = document.createElement("div");
    imgEntry.style.marginTop = "6px";
    imgEntry.textContent = `Analyzing image: ${imgUrl}`;
    imageResultsContainer.appendChild(imgEntry);

    chrome.runtime.sendMessage(
      { type: "ANALYZE_IMAGE", payload: { url: imgUrl } },
      (response) => {
        if (response && !response.error) {
          const validity = normalizeScore(response.score || 0);
          collectedScores.push(validity);
          updateBadge();

          imgEntry.innerHTML = `<img src="${imgUrl}" style="max-width:80px; max-height:50px; margin-right:6px; vertical-align:middle;"> 
                                Validity: <strong>${validity}%</strong>`;
          if (validity >= 75) imgEntry.style.color = "#0b8043";
          else if (validity >= 45) imgEntry.style.color = "#e09b00";
          else imgEntry.style.color = "#c42f2f";
        } else {
          imgEntry.textContent = `Error analyzing ${imgUrl}`;
          imgEntry.style.color = "#c42f2f";
        }
      }
    );
  });
}

// ---------------------------
// TOGGLE PANEL ON BADGE CLICK
// ---------------------------
badge.onclick = () => {
  badge.style.display = "none";
  panel.style.display = "flex";
};

setTimeout(analyzeNow, 1000);

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
