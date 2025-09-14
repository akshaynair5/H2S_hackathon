// ---------------------------
// UI PANEL
// ---------------------------
const panel = document.createElement("div");
panel.id = "trustmeter-panel";
panel.style.position = "fixed";
panel.style.bottom = "16px";
panel.style.right = "16px";
panel.style.zIndex = 2147483647;
panel.style.minWidth = "260px";
panel.style.maxWidth = "400px";
panel.style.background = "rgba(255,255,255,0.97)";
panel.style.boxShadow = "0 6px 18px rgba(0,0,0,0.18)";
panel.style.borderRadius = "10px";
panel.style.fontFamily = "Segoe UI, Roboto, Arial, sans-serif";
panel.style.padding = "10px";
panel.style.fontSize = "13px";
panel.style.color = "#111";
panel.style.display = "flex";
panel.style.flexDirection = "column";
panel.style.gap = "8px";

const header = document.createElement("div");
header.style.display = "flex";
header.style.justifyContent = "space-between";
const title = document.createElement("div");
title.textContent = "TrustMeter";
title.style.fontWeight = "600";
title.style.fontSize = "14px";
const closeBtn = document.createElement("button");
closeBtn.textContent = "×";
closeBtn.style.border = "none";
closeBtn.style.background = "transparent";
closeBtn.style.fontSize = "18px";
closeBtn.style.cursor = "pointer";
closeBtn.onclick = () => panel.remove();
header.appendChild(title);
header.appendChild(closeBtn);

const scoreRow = document.createElement("div");
scoreRow.style.display = "flex";
scoreRow.style.justifyContent = "space-between";
const scoreText = document.createElement("div");
scoreText.textContent = "Score: —";
scoreText.style.fontWeight = "700";
const refreshBtn = document.createElement("button");
refreshBtn.textContent = "Analyze";
refreshBtn.style.padding = "6px 8px";
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
// HELPER FUNCTIONS
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
  const imgs = Array.from(document.images)
    .filter(img => img.src.startsWith("http") && img.offsetWidth > 10 && img.offsetHeight > 10)
    .slice(0, maxCount)
    .map(img => img.src);
  return imgs;
}

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
}

function setResult(score, shortExplanation) {
  scoreText.textContent = `Score: ${Math.round(score*100)}%`;
  if (score >= 0.75) scoreText.style.color = "#0b8043";
  else if (score >= 0.45) scoreText.style.color = "#e09b00";
  else scoreText.style.color = "#c42f2f";

  document.getElementById("text-result").textContent =
    shortExplanation || "No explanation returned.";
}

// ---------------------------
// ANALYZE FUNCTION
// ---------------------------
function analyzeNow() {
  setWorking("Analyzing visible text & images...");

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
          const score = response.score || 0;
          const validity = Math.round(score * 100);
          imgEntry.innerHTML = `<img src="${imgUrl}" style="max-width:80px; max-height:50px; margin-right:6px; vertical-align:middle;"> 
                                Validity: <strong>${validity}%</strong>`;
          if (score >= 0.75) imgEntry.style.color = "#0b8043";
          else if (score >= 0.45) imgEntry.style.color = "#e09b00";
          else imgEntry.style.color = "#c42f2f";
        } else {
          imgEntry.textContent = `Error analyzing ${imgUrl}`;
          imgEntry.style.color = "#c42f2f";
        }
      }
    );
  });
}

setTimeout(analyzeNow, 1000);
