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
// SPINNER
// ---------------------------
const spinner = document.createElement("div");
spinner.className = "trustmeter-spinner";
Object.assign(spinner.style, {
  border: "3px solid rgba(0,0,0,0.1)",
  borderTop: "3px solid #667eea",
  borderRadius: "50%",
  width: "18px",
  height: "18px",
  animation: "spin 1s linear infinite",
  display: "inline-block",
  marginLeft: "8px",
  verticalAlign: "middle",
});

function setWorking(msg) {
  scoreText.textContent = "Score: …";
  document.getElementById("text-result").textContent = msg || "Analyzing...";

  if (!badge.contains(spinner)) {
    badge.textContent = "Trust Score: …";
    badge.appendChild(spinner);
  }
}

function stopWorking() {
  if (spinner.parentElement) spinner.remove();
}

// ---------------------------
// BADGE (collapsed)
// ---------------------------
const badge = document.createElement("div");
badge.id = "trustmeter-badge";
Object.assign(badge.style, {
  position: "fixed",
  bottom: "60px",
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
  bottom: "60px",
  zIndex: "2147483647",
  minWidth: "300px",
  maxWidth: "400px",
  background: "rgba(255, 255, 255, 0.95)",
  backdropFilter: "blur(10px)",
  boxShadow: "0 8px 20px rgba(0, 0, 0, 0.2)",
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
  if (panel.style.left !== "auto" || panel.style.top !== "auto") {
    panel.style.left = "auto";
    panel.style.top = "auto";
    panel.style.right = "24px";
    panel.style.bottom = "60px";
  }

  panel.style.display = "flex";      
  panel.style.opacity = "0";        

  requestAnimationFrame(() => {
    panel.style.opacity = "1";        
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

const subScores = document.createElement("div");
subScores.style.borderBottom = "1px solid rgba(0,0,0,0.1)";
subScores.style.paddingBottom = "8px";
Object.assign(subScores.style, {
  fontSize: "13px",
  color: "#444",
  display: "flex",
  flexDirection: "column",
  gap: "4px",
  paddingBottom: "8px"
});
subScores.id = "sub-scores";
subScores.innerHTML = `
  <div>📝 <strong>Text Score:</strong> —</div>
  <div>🖼️ <strong>Image Score:</strong> —</div>
`;

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

  const textContainer = document.getElementById("analysis-results");
  const imageContainer = document.getElementById("image-results");
  if (textContainer) textContainer.innerHTML = "";
  if (imageContainer) imageContainer.innerHTML = "";
  collectedScores = [];

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
panel.appendChild(subScores);
panel.appendChild(scoreRow);
panel.appendChild(textSection);
panel.appendChild(imageSection);
document.documentElement.appendChild(panel);

// ---------------------------
// STATE
// ---------------------------
let collectedScores = [];
let textScores = [];
let imageScores = [];

function setError(err) {
  scoreText.textContent = "Score: —";
  document.getElementById("text-result").textContent =
    "Error: " + (err || "Analysis failed or not supported.");
  badge.textContent = "Trust Score: —";
  badge.style.color = "#ffffff";
}

function setResult(score, explanation) {
  const percent = normalizeScore(score);

  textScores.push(percent);

  scoreText.textContent = `Score: ${percent}%`;
  if (percent >= 75) scoreText.style.color = "#0b8043";
  else if (percent >= 45) scoreText.style.color = "#e09b00";
  else scoreText.style.color = "#c42f2f";

  document.getElementById("text-result").textContent =
    explanation || "No explanation returned.";

  updateTextAverage();
  updateImageAverage();
  updateBadge();
}

function updateBadge() {
  const avgText =
    textScores.length > 0
      ? Math.round(textScores.reduce((a, b) => a + b, 0) / textScores.length)
      : "—";

  const avgImage =
    imageScores.length > 0
      ? Math.round(imageScores.reduce((a, b) => a + b, 0) / imageScores.length)
      : "—";

  badge.textContent = `📝 ${avgText !== "—" ? avgText + "%" : "—"} | 🖼️ ${avgImage !== "—" ? avgImage + "%" : "—"}`;

  if (avgText !== "—") {
    if (avgText >= 75) badge.style.background = "linear-gradient(135deg, #43a047, #2e7d32)";
    else if (avgText >= 45) badge.style.background = "linear-gradient(135deg, #f6ad55, #dd6b20)";
    else badge.style.background = "linear-gradient(135deg, #e53e3e, #c53030)";
  }
}

function updateTextAverage() {
  const avgText =
    textScores.length > 0
      ? Math.round(textScores.reduce((a, b) => a + b, 0) / textScores.length)
      : "—";
  document.querySelector("#sub-scores div:nth-child(1)").innerHTML =
    `📝 <strong>Text Score:</strong> ${avgText}${avgText !== "—" ? "%" : ""}`;
}

function updateImageAverage() {
  const avgImage =
    imageScores.length > 0
      ? Math.round(imageScores.reduce((a, b) => a + b, 0) / imageScores.length)
      : "—";
  document.querySelector("#sub-scores div:nth-child(2)").innerHTML =
    `🖼️ <strong>Image Score:</strong> ${avgImage}${avgImage !== "—" ? "%" : ""}`;
}


// ---------------------------
// TEXT + IMAGE HELPERS
// ---------------------------
function collectVisibleText(maxChars = 20000) {
  try {
    let text = "";

    if (typeof Readability !== "undefined") {
      const clone = document.cloneNode(true);
      const article = new Readability(clone).parse();
      if (article && article.textContent) {
        text = article.textContent.replace(/\s+/g, " ").trim().substring(0, maxChars);
        if (text.length > 300) return text;
      }
    }

    const mainContent = document.querySelector("article, main, [role='main']") || document.body;

    const removeSelectors = [
      "nav","header","footer","aside","form","button","input","textarea",
      "select","script","style","noscript","iframe","svg","canvas",
      "video","audio",".advertisement",".ads",".sponsored",".comments",
      ".related",".popup",".newsletter",".cookie",".banner",".share"
    ];

    const walker = document.createTreeWalker(mainContent, NodeFilter.SHOW_TEXT, {
      acceptNode: node => {
        const value = node.nodeValue.trim();
        if (!value) return NodeFilter.FILTER_REJECT;
        const el = node.parentElement;
        if (!el || el.closest(removeSelectors.join(","))) return NodeFilter.FILTER_REJECT;
        const style = window.getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0" || el.offsetParent === null) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });

    let node;
    const chunkSize = 5000; 
    while ((node = walker.nextNode()) && text.length < maxChars) {
      text += node.nodeValue.replace(/\s+/g, " ") + " ";
      if (text.length > chunkSize) {
        text = text.substring(0, maxChars);
      }
    }

    return text.replace(/\s+/g, " ").trim().substring(0, maxChars);
  } catch (err) {
    console.error("Error collecting text:", err);
    return "";
  }
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
  textScores = [];
  updateTextAverage();
  updateBadge();

  const visibleText = collectVisibleText(45000);
  if (!visibleText) {
    setError("No visible text found.");
    stopWorking();
    return;
  }

  chrome.runtime.sendMessage(
    { type: "ANALYZE_TEXT", payload: { text: visibleText, url: location.href } },
    response => {
      stopWorking();
      if (!response || response.error) {
        setError(response?.error || "No response from backend.");
        console.log("Text analysis failed:", response);
        return;
      }
      setResult(response.score ?? 0, response.explanation);
    }
  );
}

function analyzeImagesNow() {
  const container = document.getElementById("image-results");
  setWorking("Analyzing visible text...");

  const visibleImages = collectVisibleImages(3);
  if (visibleImages.length === 0) {
    stopWorking();
    container.textContent = "No images found on this page.";
    return;
  }

  chrome.runtime.sendMessage(
    { type: "ANALYZE_IMAGE", payload: { urls: visibleImages } },
    response => {
      stopWorking();
      if (!response || response.error) {
        container.textContent = "Image analysis failed or not supported.";
      } else {
        container.textContent = "";
      }
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
      console.log("Received TEXT_ANALYSIS_RESULT:", message.payload);
      const overall = message.payload || {};
      setResult(overall.score || 0, overall.explanation || "No explanation");
      break;


    case "IMAGE_ANALYSIS_RESULT": {
      const { url, score, explanation } = message.payload;
      const container = document.getElementById("image-results");
      if (!container) return;
      if (container.textContent.includes("No images")) container.innerHTML = "";

      if (!document.getElementById("image-analysis-header")) {
        const header = document.createElement("h3");
        header.id = "image-analysis-header";
        header.textContent = "🖼️ Image Analysis Results";
        header.style.cssText = `
          margin: 10px 0 12px;
          font-size: 15px;
          font-weight: 600;
          color: #333;
        `;
        container.prepend(header);
      }

      const validity = normalizeScore(score);
      const timestamp = new Date().toLocaleTimeString();

      imageScores.push(validity);

      const imgEntry = document.createElement("div");
      imgEntry.className = "image-result-entry";
      Object.assign(imgEntry.style, {
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        padding: "12px",
        borderRadius: "10px",
        marginBottom: "12px",
        background: "#fafafa",
        border: "1px solid rgba(0,0,0,0.08)",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
        animation: "fadeIn 0.4s ease"
      });

      let color = "#c42f2f";
      let label = "AI Generated";
      if (validity >= 75) {
        color = "#0b8043";
        label = "Likely Authentic";
      } else if (validity >= 45) {
        color = "#e09b00";
        label = "Uncertain";
      }

      imgEntry.innerHTML = `
        <div style="display:flex; align-items:center; gap:12px;">
          <img src="${url}" 
              style="width:90px; height:60px; border-radius:6px; object-fit:cover; border:1px solid rgba(0,0,0,0.1);">
          <div style="flex:1;">
            <div style="font-weight:600; color:${color}; font-size:14px;">
              ${label} (${validity}%)
            </div>
            <div style="height:6px; background:#e5e5e5; border-radius:3px; overflow:hidden; margin-top:4px;">
              <div style="width:${validity}%; background:${color}; height:100%;"></div>
            </div>
          </div>
        </div>
        <div style="margin-top:8px; font-size:13px; color:#333;">
          <strong>Explanation:</strong> ${explanation || "No details available."}
        </div>
        <div style="font-size:11px; color:#777; margin-top:4px;">Analyzed at ${timestamp}</div>
      `;

      container.appendChild(imgEntry);

      updateTextAverage();
      updateImageAverage();
      updateBadge();
      showPanel();
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
// AUTO RUN TEXT ANALYSIS
// ---------------------------
setTimeout(analyzeTextNow, 2000);

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
  const minTop = -panel.offsetHeight / 3;

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
    panel.style.bottom = "60px";
    panel.style.right = "24px";
    panel.style.transform = "translateY(0)";
  }
});

panel.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hidePanel();
  if (e.key === "Enter" || e.key === " ") {
    if (document.activeElement.tagName === "BUTTON") {
      document.activeElement.click();
      e.preventDefault();
    }
  }
});