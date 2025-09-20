// ================================
// content.js (Fixed + Improved)
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
Object.assign(badge.style, {
  position: "fixed",
  bottom: "24px",
  right: "24px",
  zIndex: "2147483647",
  padding: "10px 16px",
  borderRadius: "12px",
  background: "linear-gradient(135deg, #667eea, #764ba2)",
  color: "#ffffff",
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  fontSize: "14px",
  fontWeight: "600",
  cursor: "pointer",
  boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
  transition: "transform 0.2s ease, box-shadow 0.2s ease",
  display: "flex",
  alignItems: "center",
  gap: "8px"
});
badge.textContent = "Trust Score: —";
badge.setAttribute("role", "button");
badge.setAttribute("aria-label", "Open TrustMeter panel");

// Hover effect
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
Object.assign(panel.style, {
  position: "fixed",
  left: "auto",
  top: "auto",
  right: "24px",
  bottom: "24px",
  zIndex: "2147483647",
  minWidth: "300px",
  maxWidth: "400px",
  background: "rgba(255, 255, 255, 0.95)",
  backdropFilter: "blur(10px)",
  boxShadow: "0 8px 24px rgba(0, 0, 0, 0.2)",
  borderRadius: "16px",
  border: "1px solid rgba(255, 255, 255, 0.2)",
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  padding: "16px",
  color: "#2d3748",
  display: "none",
  flexDirection: "column",
  gap: "12px",
  maxHeight: "80vh",
  overflowY: "auto",
  transition: "opacity 0.3s ease, transform 0.3s ease, left 0.3s ease, top 0.3s ease",
  opacity: "0",
  transform: "translateY(10px)"
});
panel.setAttribute("role", "dialog");
panel.setAttribute("aria-label", "TrustMeter Analysis Panel");

// ---------------------------
// SHOW PANEL FUNCTION
// ---------------------------
function showPanel() {
  // Reset pos if dragged
  if (panel.style.left !== "auto" || panel.style.top !== "auto") {
    panel.style.left = "auto";
    panel.style.top = "auto";
    panel.style.right = "24px";
    panel.style.bottom = "24px";
  }

  panel.style.display = "flex";       // make visible
  panel.style.opacity = "0";          // start hidden

  // Let browser apply display:flex first before transition
  requestAnimationFrame(() => {
    panel.style.opacity = "1";        // transition in
    panel.style.transform = "translateY(0)";
  });
}

// ---------------------------
// HIDE PANEL FUNCTION
// ---------------------------
function hidePanel() {
  panel.style.opacity = "0";
  panel.style.transform = "translateY(10px)";
  setTimeout(() => {
    if (panel.style.opacity === "0") {
      panel.style.display = "none";
    }
  }, 300);
}

// ---------------------------
// HEADER
// ---------------------------
const header = document.createElement("div");
Object.assign(header.style, {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  paddingBottom: "8px",
  borderBottom: "1px solid rgba(0, 0, 0, 0.1)",
  cursor: "grab"
});

const title = document.createElement("div");
title.textContent = "TrustMeter";
Object.assign(title.style, {
  fontWeight: "700",
  fontSize: "16px",
  background: "linear-gradient(135deg, #667eea, #764ba2)",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent"
});

const closeBtn = document.createElement("button");
closeBtn.textContent = "×";
Object.assign(closeBtn.style, {
  border: "none",
  background: "transparent",
  fontSize: "18px",
  fontWeight: "600",
  color: "#6b7280",
  cursor: "pointer",
  transition: "color 0.2s ease"
});
closeBtn.setAttribute("aria-label", "Close TrustMeter panel");
closeBtn.addEventListener("mouseover", () => (closeBtn.style.color = "#3b82f6"));
closeBtn.addEventListener("mouseout", () => (closeBtn.style.color = "#6b7280"));
closeBtn.onclick = hidePanel;

header.appendChild(title);
header.appendChild(closeBtn);

// ---------------------------
// SCORE ROW
// ---------------------------
const scoreRow = document.createElement("div");
Object.assign(scoreRow.style, {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "8px 0"
});

const scoreText = document.createElement("div");
scoreText.textContent = "Score: —";
scoreText.style.fontWeight = "700";
scoreText.style.fontSize = "14px";

const refreshBtn = document.createElement("button");
refreshBtn.textContent = "Analyze";
Object.assign(refreshBtn.style, {
  padding: "6px 12px",
  borderRadius: "8px",
  border: "none",
  background: "linear-gradient(135deg, #667eea, #764ba2)",
  color: "#ffffff",
  fontSize: "13px",
  fontWeight: "600",
  cursor: "pointer",
  transition: "transform 0.2s ease, boxShadow 0.2s ease"
});
refreshBtn.setAttribute("aria-label", "Re-analyze content");
refreshBtn.onmouseover = () => {
  refreshBtn.style.transform = "translateY(-1px)";
  refreshBtn.style.boxShadow = "0 4px 12px rgba(102, 126, 234, 0.3)";
};
refreshBtn.onmouseout = () => {
  refreshBtn.style.transform = "translateY(0)";
  refreshBtn.style.boxShadow = "none";
};
refreshBtn.onclick = () => {
  showPanel();
  analyzeTextNow();
  analyzeImagesNow();
};
scoreRow.appendChild(scoreText);
scoreRow.appendChild(refreshBtn);

// ---------------------------
// SECTIONS
// ---------------------------
const textSection = document.createElement("div");
textSection.innerHTML = "<strong>Text Analysis:</strong><br><span id='text-result'>No analysis yet.</span>";
Object.assign(textSection.style, {
  fontSize: "13px",
  color: "#2d3748",
  lineHeight: "1.5",
  padding: "8px 0"
});

const imageSection = document.createElement("div");
imageSection.innerHTML = "<strong>Image Analysis:</strong><br><div id='image-results'>No images analyzed.</div>";
Object.assign(imageSection.style, {
  fontSize: "13px",
  color: "#2d3748",
  lineHeight: "1.5",
  padding: "8px 0"
});

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
// UI HELPERS
// ---------------------------
function setWorking(msg) {
  scoreText.textContent = "Score: …";
  document.getElementById("text-result").textContent = msg || "Analyzing...";
}

function setError(err) {
  scoreText.textContent = "Score: —";
  document.getElementById("text-result").textContent =
    "Error: " + (err || "Analysis failed or not supported.");
  badge.textContent = "Trust Score: —";
  badge.style.color = "#ffffff";
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
  badge.style.color = "#ffffff";
}

// ---------------------------
// TEXT + IMAGE HELPERS
// ---------------------------
function collectVisibleText(maxChars = 20000) {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode: node => {
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
    .filter(
      img =>
        img.src.startsWith("http") &&
        img.offsetWidth > 10 &&
        img.offsetHeight > 10
    )
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
    response => {
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
    container.textContent = "No images found on this page.";
    return;
  }

  chrome.runtime.sendMessage(
    { type: "ANALYZE_IMAGE", payload: { urls: visibleImages } },
    response => {
      container.innerHTML = "";
      if (!response || response.error) {
        container.textContent = "Image analysis failed or not supported.";
        return;
      }

      response.forEach((imgResult, idx) => {
        const imgUrl = visibleImages[idx];
        const validity = normalizeScore(imgResult.score);
        const imgEntry = document.createElement("div");
        Object.assign(imgEntry.style, {
          display: "flex",
          alignItems: "center",
          padding: "8px 0",
          borderTop: idx > 0 ? "1px solid rgba(0,0,0,0.1)" : "none"
        });
        imgEntry.innerHTML = `
          <img src="${imgUrl}" style="max-width:80px; max-height:50px; margin-right:12px; border-radius:4px; object-fit:cover;">
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
// MESSAGE HANDLER
// ---------------------------
chrome.runtime.onMessage.addListener(message => {
  if (!message?.type) return;

  switch (message.type) {
    case "TEXT_ANALYSIS_RESULT":
      setResult(message.payload.score, message.payload.explanation);
      break;

    case "IMAGE_ANALYSIS_RESULT": {
      const { url, score } = message.payload;
      const container = document.getElementById("image-results");
      if (container.textContent.includes("No images")) container.innerHTML = "";

      const validity = normalizeScore(score);
      const imgEntry = document.createElement("div");
      Object.assign(imgEntry.style, {
        display: "flex",
        alignItems: "center",
        padding: "8px 0",
        borderTop:
          container.children.length > 0 ? "1px solid rgba(0,0,0,0.1)" : "none"
      });
            imgEntry.innerHTML = `
        <img src="${url}" style="max-width:80px; max-height:50px; margin-right:12px; border-radius:4px; object-fit:cover;">
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
      setError("Analysis failed or not supported on this page.");
      break;

    case "EXPAND_PANEL_UI":
      showPanel();
      break;
  }
});

// ---------------------------
// AUTO RUN TEXT ANALYSIS (once)
// ---------------------------
setTimeout(analyzeTextNow, 1000);

// ---------------------------
// TOGGLE PANEL (badge click)
// ---------------------------
badge.onclick = () => {
  const isVisible = getComputedStyle(panel).display !== "none" && panel.style.opacity === "1";
  if (isVisible) hidePanel();
  else showPanel();
};

// ---------------------------
// ESC KEY CLOSE SUPPORT
// ---------------------------
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && panel.style.display === "flex") {
    hidePanel();
  }
});

// ---------------------------
// DRAGGABLE + AUTO-HIDE
// ---------------------------
let isDragging = false;
let dragOffsetX = 0;
let dragOffsetY = 0;

header.addEventListener("mousedown", (e) => {
  isDragging = true;
  dragOffsetX = e.clientX - panel.getBoundingClientRect().left;
  dragOffsetY = e.clientY - panel.getBoundingClientRect().top;
  header.style.cursor = "grabbing";
});

document.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  panel.style.transition = "none";
  let left = e.clientX - dragOffsetX;
  let top = e.clientY - dragOffsetY;

  const maxLeft = window.innerWidth - panel.offsetWidth / 3;
  const maxTop = window.innerHeight - panel.offsetHeight / 3;
  const minLeft = -panel.offsetWidth * 2 / 3;
  const minTop = 0;

  left = Math.min(Math.max(left, minLeft), maxLeft);
  top = Math.min(Math.max(top, minTop), maxTop);

  panel.style.left = left + "px";
  panel.style.top = top + "px";
  panel.style.bottom = "auto";
  panel.style.right = "auto";
});

document.addEventListener("mouseup", () => {
  if (!isDragging) return;
  isDragging = false;
  header.style.cursor = "grab";
  panel.style.transition =
    "opacity 0.3s ease, transform 0.3s ease, left 0.3s ease, top 0.3s ease";

  const rect = panel.getBoundingClientRect();
  if (rect.right > window.innerWidth) {
    panel.style.left = window.innerWidth - panel.offsetWidth / 4 + "px";
  } else if (rect.left < 0) {
    panel.style.left = -panel.offsetWidth / 4 + "px";
  }
});

// ---------------------------
// RESTORE CLICK WHEN PARTIALLY HIDDEN
// ---------------------------
panel.addEventListener("click", () => {
  const rect = panel.getBoundingClientRect();
  if (rect.left < 0 || rect.right > window.innerWidth) {
    panel.style.left = "auto";
    panel.style.top = "auto";
    panel.style.bottom = "24px";
    panel.style.right = "24px";
    panel.style.transform = "translateY(0)";
  }
});