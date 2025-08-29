// content.js
// Runs on pages defined in manifest. Extracts visible text from the viewport,
// draws a small floating panel, and asks the background/service_worker to query backend LLM.

(function() {
  // Avoid running twice
  if (window.__trustmeter_loaded) return;
  window.__trustmeter_loaded = true;

  // Utility: check if element is visible in viewport
  function isElementVisibleInViewport(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return false;
    // require some overlap with viewport
    const vw = (window.innerWidth || document.documentElement.clientWidth);
    const vh = (window.innerHeight || document.documentElement.clientHeight);
    return !(rect.bottom < 0 || rect.right < 0 || rect.top > vh || rect.left > vw);
  }

  // Walk and collect visible text nodes within viewport
  function collectVisibleText(limitChars = 40000) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
      acceptNode: function(node) {
        if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
        // avoid script/style/comments
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        const tag = parent.tagName.toLowerCase();
        if (["script","style","noscript","iframe"].includes(tag)) return NodeFilter.FILTER_REJECT;
        // trim whitespace-only
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    });

    let text = "";
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const parent = node.parentElement;
      if (!isElementVisibleInViewport(parent)) continue;
      // avoid text inside inputs
      if (["input","textarea"].includes(parent.tagName.toLowerCase())) continue;
      text += node.nodeValue.trim() + " ";
      if (text.length > limitChars) break;
    }
    return text.trim();
  }

  // Create floating panel
  const panel = document.createElement("div");
  panel.id = "trustmeter-panel";
  panel.style.position = "fixed";
  panel.style.bottom = "16px";
  panel.style.right = "16px";
  panel.style.zIndex = 2147483647; // maximum z-index
  panel.style.minWidth = "220px";
  panel.style.maxWidth = "380px";
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
  panel.style.backdropFilter = "blur(4px)";
  panel.style.transition = "transform 0.12s ease, opacity 0.12s ease";
  panel.style.cursor = "default";

  // Header row
  const header = document.createElement("div");
  header.style.display = "flex";
  header.style.justifyContent = "space-between";
  header.style.alignItems = "center";

  const title = document.createElement("div");
  title.textContent = "TrustMeter";
  title.style.fontWeight = "600";
  title.style.fontSize = "14px";

  const closeBtn = document.createElement("button");
  closeBtn.textContent = "×";
  closeBtn.title = "Close";
  closeBtn.style.border = "none";
  closeBtn.style.background = "transparent";
  closeBtn.style.fontSize = "18px";
  closeBtn.style.cursor = "pointer";
  closeBtn.style.lineHeight = "1";
  closeBtn.onclick = () => panel.remove();

  header.appendChild(title);
  header.appendChild(closeBtn);

  // Score display
  const scoreRow = document.createElement("div");
  scoreRow.style.display = "flex";
  scoreRow.style.alignItems = "center";
  scoreRow.style.justifyContent = "space-between";

  const scoreText = document.createElement("div");
  scoreText.textContent = "Score: —";
  scoreText.style.fontWeight = "700";

  const refreshBtn = document.createElement("button");
  refreshBtn.textContent = "Analyze";
  refreshBtn.style.padding = "6px 8px";
  refreshBtn.style.borderRadius = "6px";
  refreshBtn.style.border = "1px solid rgba(0,0,0,0.08)";
  refreshBtn.style.background = "#fff";
  refreshBtn.style.cursor = "pointer";
  refreshBtn.onclick = analyzeNow;

  scoreRow.appendChild(scoreText);
  scoreRow.appendChild(refreshBtn);

  // Explanation
  const explanation = document.createElement("div");
  explanation.textContent = "No analysis yet. Click Analyze to send visible text to your configured backend LLM.";
  explanation.style.fontSize = "12px";
  explanation.style.color = "#333";
  explanation.style.maxHeight = "160px";
  explanation.style.overflow = "auto";

  // Footer with small link
  const footer = document.createElement("div");
  footer.style.fontSize = "11px";
  footer.style.color = "#666";
  footer.style.display = "flex";
  footer.style.justifyContent = "space-between";
  footer.style.alignItems = "center";

  const powered = document.createElement("div");
  powered.textContent = "LLM-powered";
  const detailsLink = document.createElement("a");
  detailsLink.href = "#";
  detailsLink.textContent = "Options";
  detailsLink.style.fontSize = "12px";
  detailsLink.onclick = (e) => {
    e.preventDefault();
    // open extension options page
    try { chrome.runtime.openOptionsPage(); } catch (err) { console.warn(err); }
  };

  footer.appendChild(powered);
  footer.appendChild(detailsLink);

  // Assemble
  panel.appendChild(header);
  panel.appendChild(scoreRow);
  panel.appendChild(explanation);
  panel.appendChild(footer);
  document.documentElement.appendChild(panel);

  // Show temporary "working" state
  let lastScore = null;
  function setWorking() {
    scoreText.textContent = "Score: …";
    explanation.textContent = "Analyzing visible text...";
    panel.style.opacity = "0.98";
  }

  function setError(err) {
    scoreText.textContent = "Score: —";
    explanation.textContent = "Error: " + (err && err.message ? err.message : String(err));
  }

  function setResult(score, shortExplanation) {
    lastScore = score;
    scoreText.textContent = `Score: ${Math.round(score*100)}%`;
    // simple color hint
    if (score >= 0.75) scoreText.style.color = "#0b8043";
    else if (score >= 0.45) scoreText.style.color = "#e09b00";
    else scoreText.style.color = "#c42f2f";
    explanation.textContent = shortExplanation || "No explanation returned.";
  }
  let analyseScheduled = false;
  function scheduleAnalyze() {
    if (analyseScheduled) return;
    analyseScheduled = true;
    setTimeout(() => { analyseScheduled = false; analyzeNow(); }, 900);
  }
  window.addEventListener("scroll", scheduleAnalyze, { passive: true });
  window.addEventListener("resize", scheduleAnalyze);

  function analyzeNow() {
    setWorking();
    const visibleText = collectVisibleText(45000);
    if (!visibleText) {
      setError("No visible text in viewport.");
      return;
    }

    chrome.runtime.sendMessage({ type: "ANALYZE_TEXT", payload: { text: visibleText, url: location.href } }, function(response) {
      if (!response) {
        setError("No response (maybe no backend configured). Open Options.");
        return;
      }
      if (response.error) {
        setError(response.error);
        return;
      }

      setResult(typeof response.score === "number" ? response.score : 0, response.explanation || "");
    });
  }

  setTimeout(analyzeNow, 800);
})();
