// ---------------------------
// SESSION MANAGEMENT
// ---------------------------
let sessionId = localStorage.getItem('trustmeter_session_id');
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem('trustmeter_session_id', sessionId);
}
console.log("TrustMeter Session ID:", sessionId);

// ---------------------------
// CANCEL SESSION ON EXIT
// ---------------------------
function cancelSession() {
  if (!sessionId) return;
  
  console.log("Cancelling session:", sessionId);
  const blob = new Blob(
    [JSON.stringify({ session_id: sessionId })],
    { type: 'application/json' }
  );
  
  const sent = navigator.sendBeacon('http://localhost:5000/cancel_session', blob);
  if (!sent) {
    fetch('http://localhost:5000/cancel_session', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': sessionId
      },
      body: JSON.stringify({ session_id: sessionId }),
      keepalive: true
    }).catch(err => console.log('Session cancel failed:', err));
  }
}

window.addEventListener('beforeunload', cancelSession);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    cancelSession();
  }
});

if (typeof chrome !== 'undefined' && chrome.runtime) {
  chrome.runtime.onSuspend?.addListener(() => {
    cancelSession();
  });
}

function normalizeScore(raw) {
  let s = raw || 0;
  if (s <= 1) s = Math.round(s * 100);
  else s = Math.round(s);
  return Math.max(0, Math.min(100, s));
}

// ---------------------------
// SPINNER (MODERN GRADIENT)
// ---------------------------
const spinner = document.createElement("div");
spinner.className = "trustmeter-spinner";
Object.assign(spinner.style, {
  border: "3px solid rgba(102, 126, 234, 0.1)",
  borderTop: "3px solid #667eea",
  borderRadius: "50%",
  width: "18px",
  height: "18px",
  animation: "spin 0.8s linear infinite",
  display: "inline-block",
  marginLeft: "8px",
  verticalAlign: "middle",
});

// Add keyframe animation
if (!document.getElementById('trustmeter-keyframes')) {
  const style = document.createElement('style');
  style.id = 'trustmeter-keyframes';
  style.textContent = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.7; transform: scale(1.1); }
    }
  `;
  document.head.appendChild(style);
}

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
// BADGE (MODERN GLASSMORPHISM)
// ---------------------------
const badge = document.createElement("div");
badge.id = "trustmeter-badge";
Object.assign(badge.style, {
  position: "fixed",
  bottom: "60px",
  right: "24px",
  zIndex: "2147483647",
  padding: "12px 18px",
  borderRadius: "16px",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  color: "#ffffff",
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  fontSize: "14px",
  fontWeight: "600",
  cursor: "pointer",
  boxShadow: "0 8px 24px rgba(102, 126, 234, 0.4), 0 2px 8px rgba(0, 0, 0, 0.1)",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  display: "flex",
  alignItems: "center",
  gap: "8px",
  backdropFilter: "blur(10px)",
  border: "1px solid rgba(255, 255, 255, 0.2)"
});
badge.textContent = "Trust Score: —";
badge.setAttribute("role", "button");
badge.setAttribute("aria-label", "Open TrustMeter panel");

badge.addEventListener("mouseover", () => {
  badge.style.transform = "translateY(-3px)";
  badge.style.boxShadow = "0 12px 32px rgba(102, 126, 234, 0.5), 0 4px 12px rgba(0, 0, 0, 0.15)";
});
badge.addEventListener("mouseout", () => {
  badge.style.transform = "translateY(0)";
  badge.style.boxShadow = "0 8px 24px rgba(102, 126, 234, 0.4), 0 2px 8px rgba(0, 0, 0, 0.1)";
});
document.documentElement.appendChild(badge);

// ---------------------------
// PANEL (MODERN GLASSMORPHISM)
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
  minWidth: "320px",
  maxWidth: "420px",
  background: "rgba(255, 255, 255, 0.95)",
  backdropFilter: "blur(20px)",
  WebkitBackdropFilter: "blur(20px)",
  boxShadow: "0 12px 48px rgba(31, 38, 135, 0.2), 0 4px 16px rgba(0, 0, 0, 0.12)",
  borderRadius: "24px",
  border: "1px solid rgba(255, 255, 255, 0.3)",
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  padding: "24px",
  color: "#334155",
  display: "none",
  flexDirection: "column",
  gap: "16px",
  maxHeight: "80vh",
  overflowY: "auto",
  transition: "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)",
  opacity: "0",
  transform: "translateY(10px)"
});
panel.setAttribute("role", "dialog");
panel.setAttribute("aria-label", "TrustMeter Analysis Panel");

// Custom scrollbar
const panelStyle = document.createElement('style');
panelStyle.textContent = `
  #trustmeter-panel::-webkit-scrollbar {
    width: 6px;
  }
  #trustmeter-panel::-webkit-scrollbar-track {
    background: rgba(241, 245, 249, 0.5);
    border-radius: 3px;
  }
  #trustmeter-panel::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    border-radius: 3px;
  }
  #trustmeter-panel::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
  }
`;
document.head.appendChild(panelStyle);

// ---------------------------
// SHOW PANEL FUNCTION
// ---------------------------
function showPanel() {
  panel.style.display = "flex";
  panel.style.transition = "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)";
  requestAnimationFrame(() => {
    panel.style.opacity = "1";
    panel.style.transform = "translateY(0)";
  });
}

// ---------------------------
// HIDE PANEL FUNCTION
// ---------------------------
function hidePanel() {
  panel.style.transition = "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)";
  panel.style.opacity = "0";
  panel.style.transform = "translateY(10px)";
  setTimeout(() => {
    panel.style.display = "none";
  }, 300);
}

// ---------------------------
// HEADER (GRADIENT TEXT)
// ---------------------------
const header = document.createElement("div");
Object.assign(header.style, {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  paddingBottom: "12px",
  borderBottom: "1px solid rgba(226, 232, 240, 0.8)",
  cursor: "grab"
});

const title = document.createElement("div");
title.textContent = "✨ TrustMeter";
Object.assign(title.style, {
  fontWeight: "800",
  fontSize: "18px",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  WebkitBackgroundClip: "text",
  WebkitTextFillColor: "transparent",
  backgroundClip: "text",
  letterSpacing: "-0.5px"
});

const closeBtn = document.createElement("button");
closeBtn.textContent = "×";
Object.assign(closeBtn.style, {
  border: "none",
  background: "rgba(102, 126, 234, 0.1)",
  fontSize: "22px",
  fontWeight: "600",
  color: "#667eea",
  cursor: "pointer",
  transition: "all 0.2s ease",
  width: "32px",
  height: "32px",
  borderRadius: "10px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center"
});
closeBtn.setAttribute("aria-label", "Close TrustMeter panel");
closeBtn.addEventListener("mouseover", () => {
  closeBtn.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
  closeBtn.style.color = "#ffffff";
  closeBtn.style.transform = "scale(1.05)";
});
closeBtn.addEventListener("mouseout", () => {
  closeBtn.style.background = "rgba(102, 126, 234, 0.1)";
  closeBtn.style.color = "#667eea";
  closeBtn.style.transform = "scale(1)";
});
closeBtn.onclick = hidePanel;

header.appendChild(title);
header.appendChild(closeBtn);

// ---------------------------
// SCORE ROW (MODERN CARDS)
// ---------------------------
const scoreRow = document.createElement("div");
Object.assign(scoreRow.style, {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "8px 0"
});

const subScores = document.createElement("div");
Object.assign(subScores.style, {
  fontSize: "13px",
  color: "#475569",
  display: "flex",
  flexDirection: "column",
  gap: "8px",
  paddingBottom: "12px",
  borderBottom: "1px solid rgba(226, 232, 240, 0.8)",
  width: "100%",
  fontWeight: "500"
});
subScores.id = "sub-scores";
subScores.innerHTML = `
  <div style="padding: 10px 14px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(226, 232, 240, 0.6);">
    📝 <strong>Text Score:</strong> <span style="float: right; color: #667eea; font-weight: 700;">—</span>
  </div>
  <div style="padding: 10px 14px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(226, 232, 240, 0.6);">
    🖼️ <strong>Image Score:</strong> <span style="float: right; color: #764ba2; font-weight: 700;">—</span>
  </div>
`;

const scoreText = document.createElement("div");
scoreText.textContent = "Score: —";
Object.assign(scoreText.style, {
  fontWeight: "700",
  fontSize: "16px",
  padding: "12px 16px",
  borderRadius: "12px",
  background: "linear-gradient(135deg, rgba(248, 250, 252, 0.9) 0%, rgba(241, 245, 249, 0.9) 100%)",
  border: "1px solid rgba(226, 232, 240, 0.8)",
  transition: "all 0.3s ease"
});

const refreshBtn = document.createElement("button");
refreshBtn.textContent = "🔄 Analyze";
Object.assign(refreshBtn.style, {
  padding: "10px 18px",
  borderRadius: "12px",
  border: "none",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  color: "#ffffff",
  fontSize: "14px",
  fontWeight: "600",
  cursor: "pointer",
  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
  boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)"
});
refreshBtn.setAttribute("aria-label", "Re-analyze content");
refreshBtn.onmouseover = () => {
  refreshBtn.style.transform = "translateY(-2px)";
  refreshBtn.style.boxShadow = "0 8px 25px rgba(102, 126, 234, 0.5)";
};
refreshBtn.onmouseout = () => {
  refreshBtn.style.transform = "translateY(0)";
  refreshBtn.style.boxShadow = "0 4px 15px rgba(102, 126, 234, 0.4)";
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
// SECTIONS (MODERN DESIGN)
// ---------------------------
const textSection = document.createElement("div");
textSection.innerHTML = "<strong style='color: #334155; font-size: 14px;'>📝 Text Analysis:</strong><br><span id='text-result' style='color: #64748b; font-size: 13px; line-height: 1.6;'>No analysis yet.</span>";
Object.assign(textSection.style, {
  fontSize: "13px",
  color: "#334155",
  lineHeight: "1.6",
  padding: "14px",
  background: "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)",
  borderRadius: "14px",
  border: "1px solid rgba(226, 232, 240, 0.8)"
});

const imageSection = document.createElement("div");
imageSection.innerHTML = "<strong style='color: #334155; font-size: 14px;'>🖼️ Image Analysis:</strong><br><div id='image-results' style='color: #64748b; font-size: 13px; line-height: 1.6; margin-top: 8px;'>No images analyzed.</div>";
Object.assign(imageSection.style, {
  fontSize: "13px",
  color: "#334155",
  lineHeight: "1.6",
  padding: "14px",
  background: "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)",
  borderRadius: "14px",
  border: "1px solid rgba(226, 232, 240, 0.8)"
});

const sourcesSection = document.createElement("div");
// ---------------------------
// SOURCES SECTION (MODERN SCROLLABLE)
// ---------------------------
sourcesSection.id = "sources-section";
Object.assign(sourcesSection.style, {
  fontSize: "13px",
  color: "#334155",
  lineHeight: "1.6",
  padding: "0",
  background: "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)",
  borderRadius: "14px",
  border: "1px solid rgba(226, 232, 240, 0.8)",
  maxHeight: "300px",
  overflowY: "auto",
  display: "none" // Hidden initially
});

const sourcesHeader = document.createElement("div");
Object.assign(sourcesHeader.style, {
  position: "sticky",
  top: "0",
  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
  padding: "12px 14px",
  borderRadius: "14px 14px 0 0",
  fontWeight: "700",
  fontSize: "14px",
  color: "#ffffff",
  display: "flex",
  alignItems: "center",
  gap: "8px",
  zIndex: "10",
  boxShadow: "0 2px 8px rgba(102, 126, 234, 0.2)"
});
sourcesHeader.innerHTML = "🔗 Verified Sources";

const sourcesContent = document.createElement("div");
sourcesContent.id = "sources-content";
Object.assign(sourcesContent.style, {
  padding: "14px",
  display: "flex",
  flexDirection: "column",
  gap: "10px"
});

sourcesSection.appendChild(sourcesHeader);
sourcesSection.appendChild(sourcesContent);
panel.appendChild(header);
panel.appendChild(subScores);
panel.appendChild(scoreRow);
panel.appendChild(textSection);
panel.appendChild(sourcesSection);
panel.appendChild(imageSection);

// Custom scrollbar for sources
const sourcesScrollStyle = document.createElement('style');
sourcesScrollStyle.textContent = `
  #sources-section::-webkit-scrollbar {
    width: 6px;
  }
  #sources-section::-webkit-scrollbar-track {
    background: rgba(241, 245, 249, 0.5);
    border-radius: 3px;
  }
  #sources-section::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    border-radius: 3px;
  }
  #sources-section::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
  }
`;
document.head.appendChild(sourcesScrollStyle);

// Helper function to display sources
function displaySources(sources) {
  const sourcesContent = document.getElementById("sources-content");
  const sourcesSection = document.getElementById("sources-section");
  
  if (!sources || (!sources.corroboration?.length && !sources.fact_checks?.length)) {
    sourcesSection.style.display = "none";
    return;
  }

  sourcesContent.innerHTML = ""; // Clear existing
  sourcesSection.style.display = "block";

  // Display corroboration sources (Google search results)
  if (sources.corroboration && sources.corroboration.length > 0) {
    const corrobSection = document.createElement("div");
    corrobSection.style.marginBottom = "16px";

    const corrobTitle = document.createElement("div");
    corrobTitle.textContent = "📰 News Articles";
    Object.assign(corrobTitle.style, {
      fontWeight: "700",
      fontSize: "13px",
      color: "#334155",
      marginBottom: "10px",
      display: "flex",
      alignItems: "center",
      gap: "6px"
    });
    corrobSection.appendChild(corrobTitle);

    sources.corroboration.forEach((source, idx) => {
      const sourceCard = document.createElement("a");
      sourceCard.href = source.link;
      sourceCard.target = "_blank";
      sourceCard.rel = "noopener noreferrer";
      
      Object.assign(sourceCard.style, {
        display: "block",
        padding: "12px",
        background: "linear-gradient(135deg, #ffffff 0%, #fafbfc 100%)",
        borderRadius: "10px",
        border: "1px solid rgba(226, 232, 240, 0.6)",
        marginBottom: "8px",
        textDecoration: "none",
        color: "inherit",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        cursor: "pointer",
        position: "relative",
        overflow: "hidden"
      });

      // Score indicator color
      const score = source.evidence_score || source.similarity || 0;
      const scorePercent = Math.round(score * 100);
      let scoreColor = "#15803d";
      if (scorePercent < 75) scoreColor = "#ca8a04";
      if (scorePercent < 45) scoreColor = "#b91c1c";

      sourceCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
          <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 600; font-size: 13px; color: #667eea; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
              ${source.title || "Untitled"}
            </div>
            <div style="font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: rgba(102, 126, 234, 0.1); border-radius: 6px; font-weight: 600;">
                🌐 ${source.domain || new URL(source.link).hostname}
              </span>
              ${source.is_new_domain ? '<span style="padding: 2px 8px; background: rgba(234, 179, 8, 0.1); color: #ca8a04; border-radius: 6px; font-size: 10px; font-weight: 600;">NEW</span>' : ''}
            </div>
          </div>
          <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 4px; margin-left: 12px;">
            <div style="font-size: 11px; font-weight: 700; color: ${scoreColor}; background: ${scoreColor}15; padding: 4px 8px; border-radius: 6px;">
              ${scorePercent}%
            </div>
            ${source.domain_score ? `<div style="font-size: 9px; color: #64748b;">Trust: ${Math.round(source.domain_score * 100)}%</div>` : ''}
          </div>
        </div>
        <div style="font-size: 12px; color: #64748b; line-height: 1.5; margin-bottom: 6px;">
          ${source.snippet ? source.snippet.substring(0, 150) + (source.snippet.length > 150 ? '...' : '') : 'No preview available'}
        </div>
        <div style="height: 3px; background: rgba(226, 232, 240, 0.5); border-radius: 2px; overflow: hidden;">
          <div style="width: ${scorePercent}%; height: 100%; background: ${scoreColor}; transition: width 0.3s ease; border-radius: 2px;"></div>
        </div>
      `;

      sourceCard.addEventListener("mouseenter", () => {
        sourceCard.style.transform = "translateX(4px)";
        sourceCard.style.boxShadow = "0 4px 12px rgba(102, 126, 234, 0.2)";
        sourceCard.style.borderColor = "rgba(102, 126, 234, 0.4)";
      });

      sourceCard.addEventListener("mouseleave", () => {
        sourceCard.style.transform = "translateX(0)";
        sourceCard.style.boxShadow = "none";
        sourceCard.style.borderColor = "rgba(226, 232, 240, 0.6)";
      });

      corrobSection.appendChild(sourceCard);
    });

    sourcesContent.appendChild(corrobSection);
  }

  if (sources.fact_checks && sources.fact_checks.length > 0) {
    const factCheckSection = document.createElement("div");

    const factCheckTitle = document.createElement("div");
    factCheckTitle.textContent = "Sources and Fact-Checks";
    Object.assign(factCheckTitle.style, {
      fontWeight: "700",
      fontSize: "13px",
      color: "#334155",
      marginBottom: "10px",
      display: "flex",
      alignItems: "center",
      gap: "6px"
    });
    factCheckSection.appendChild(factCheckTitle);

    sources.fact_checks.forEach((fc, idx) => {
      let ratingColor = "#64748b";
      let ratingBg = "rgba(100, 116, 139, 0.1)";
      
      if (fc.rating_category === "false") {
        ratingColor = "#b91c1c";
        ratingBg = "rgba(239, 68, 68, 0.1)";
      } else if (fc.rating_category === "true") {
        ratingColor = "#15803d";
        ratingBg = "rgba(34, 197, 94, 0.1)";
      } else if (fc.rating_category === "mixed") {
        ratingColor = "#ca8a04";
        ratingBg = "rgba(234, 179, 8, 0.1)";
      }

      const factCheckCard = document.createElement("a");
      factCheckCard.href = fc.link;
      factCheckCard.target = "_blank";
      factCheckCard.rel = "noopener noreferrer";
      
      Object.assign(factCheckCard.style, {
        display: "block",
        padding: "12px",
        background: "linear-gradient(135deg, #ffffff 0%, #fafbfc 100%)",
        borderRadius: "10px",
        border: `1px solid ${ratingColor}40`,
        borderLeft: `4px solid ${ratingColor}`,
        marginBottom: "8px",
        textDecoration: "none",
        color: "inherit",
        transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
        cursor: "pointer"
      });

      factCheckCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
          <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 600; font-size: 12px; color: #334155; margin-bottom: 4px;">
              ${fc.publisher || "Unknown Publisher"}
            </div>
            <div style="font-size: 11px; color: #64748b; line-height: 1.4;">
              ${fc.title || fc.claim || "No details available"}
            </div>
          </div>
          <div style="margin-left: 12px; white-space: nowrap;">
            <span style="font-size: 10px; font-weight: 700; color: ${ratingColor}; background: ${ratingBg}; padding: 4px 10px; border-radius: 8px; text-transform: uppercase; letter-spacing: 0.5px;">
              ${fc.rating || fc.rating_category || "Unknown"}
            </span>
          </div>
        </div>
      `;

      factCheckCard.addEventListener("mouseenter", () => {
        factCheckCard.style.transform = "translateX(4px)";
        factCheckCard.style.boxShadow = `0 4px 12px ${ratingColor}20`;
      });

      factCheckCard.addEventListener("mouseleave", () => {
        factCheckCard.style.transform = "translateX(0)";
        factCheckCard.style.boxShadow = "none";
      });

      factCheckSection.appendChild(factCheckCard);
    });

    sourcesContent.appendChild(factCheckSection);
  }

  // Add empty state if no sources
  if (sourcesContent.children.length === 0) {
    sourcesContent.innerHTML = `
      <div style="text-align: center; padding: 20px; color: #94a3b8; font-size: 13px;">
        <div style="font-size: 32px; margin-bottom: 8px; opacity: 0.5;">🔍</div>
        No verified sources found for this content.
      </div>
    `;
  }
}

document.documentElement.appendChild(panel);

// ---------------------------
// STATE
// ---------------------------
let collectedScores = [];
let textScores = [];
let imageScores = [];

function setError(err) {
  scoreText.textContent = "Score: —";
  scoreText.style.background = "linear-gradient(135deg, rgba(254, 202, 202, 0.3) 0%, rgba(252, 165, 165, 0.2) 100%)";
  scoreText.style.border = "1px solid rgba(239, 68, 68, 0.2)";
  scoreText.style.color = "#b91c1c";
  
  document.getElementById("text-result").textContent =
    "Error: " + (err || "Analysis failed or not supported.");
  badge.textContent = "Trust Score: —";
  badge.style.background = "linear-gradient(135deg, #e53e3e 0%, #c53030 100%)";
}

function setResult(score, explanation) {
  const percent = normalizeScore(score);

  textScores.push(percent);

  scoreText.textContent = `Score: ${percent}%`;
  
  if (percent >= 75) {
    scoreText.style.color = "#15803d";
    scoreText.style.background = "linear-gradient(135deg, rgba(187, 247, 208, 0.3) 0%, rgba(134, 239, 172, 0.2) 100%)";
    scoreText.style.border = "1px solid rgba(34, 197, 94, 0.2)";
  } else if (percent >= 45) {
    scoreText.style.color = "#ca8a04";
    scoreText.style.background = "linear-gradient(135deg, rgba(254, 240, 138, 0.3) 0%, rgba(253, 224, 71, 0.2) 100%)";
    scoreText.style.border = "1px solid rgba(234, 179, 8, 0.2)";
  } else {
    scoreText.style.color = "#b91c1c";
    scoreText.style.background = "linear-gradient(135deg, rgba(254, 202, 202, 0.3) 0%, rgba(252, 165, 165, 0.2) 100%)";
    scoreText.style.border = "1px solid rgba(239, 68, 68, 0.2)";
  }

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
    if (avgText >= 75) badge.style.background = "linear-gradient(135deg, #43a047 0%, #2e7d32 100%)";
    else if (avgText >= 45) badge.style.background = "linear-gradient(135deg, #f6ad55 0%, #dd6b20 100%)";
    else badge.style.background = "linear-gradient(135deg, #e53e3e 0%, #c53030 100%)";
  }
}

function updateTextAverage() {
  const avgText =
    textScores.length > 0
      ? Math.round(textScores.reduce((a, b) => a + b, 0) / textScores.length)
      : "—";
  
  const textScoreCard = document.querySelector("#sub-scores div:nth-child(1)");
  const scoreSpan = textScoreCard.querySelector("span");
  scoreSpan.textContent = avgText !== "—" ? `${avgText}%` : "—";
  
  if (avgText !== "—") {
    if (avgText >= 75) scoreSpan.style.color = "#15803d";
    else if (avgText >= 45) scoreSpan.style.color = "#ca8a04";
    else scoreSpan.style.color = "#b91c1c";
  }
}

function updateImageAverage() {
  const avgImage =
    imageScores.length > 0
      ? Math.round(imageScores.reduce((a, b) => a + b, 0) / imageScores.length)
      : "—";
  
  const imageScoreCard = document.querySelector("#sub-scores div:nth-child(2)");
  const scoreSpan = imageScoreCard.querySelector("span");
  scoreSpan.textContent = avgImage !== "—" ? `${avgImage}%` : "—";
  
  if (avgImage !== "—") {
    if (avgImage >= 75) scoreSpan.style.color = "#15803d";
    else if (avgImage >= 45) scoreSpan.style.color = "#ca8a04";
    else scoreSpan.style.color = "#b91c1c";
  }
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
    { 
      type: "ANALYZE_TEXT", 
      payload: { 
        text: visibleText, 
        url: location.href,
        session_id: sessionId  
      } 
    },
    response => {
      stopWorking();
      if (!response || response.error) {
        setError(response?.error || "No response from backend.");
        console.log("Text analysis failed:", response);
        return;
      }
      setResult(response.score ?? 0, response.explanation);

      if (response.session_id) {
        console.log("Analysis completed for session:", response.session_id);
      }
    }
  );
}

function analyzeImagesNow() {
  const container = document.getElementById("image-results");
  setWorking("Analyzing images...");

  const visibleImages = collectVisibleImages(3);
  if (visibleImages.length === 0) {
    stopWorking();
    container.textContent = "No images found on this page.";
    return;
  }
  chrome.runtime.sendMessage(
    { 
      type: "ANALYZE_IMAGE", 
      payload: { 
        urls: visibleImages,
        session_id: sessionId  
      } 
    },
    response => {
      stopWorking();
      if (!response || response.error) {
        container.textContent = "Image analysis failed or not supported.";
      } else {
        container.textContent = "";
      }
      if (response.session_id) {
        console.log("Image analysis completed for session:", response.session_id);
      }
    }
  );
}

chrome.runtime.onMessage.addListener(message => {
  if (!message?.type) return;

  switch (message.type) {
    case "TEXT_ANALYSIS_RESULT":
      console.log("Received TEXT_ANALYSIS_RESULT:", message.payload);
      const overall = message.payload || {};
      setResult(overall.score || 0, overall.explanation || "No explanation");

      if (overall.session_id && overall.session_id !== sessionId) {
        console.warn("Received result for different session:", overall.session_id);
      }

      if (overall.sources) {
        displaySources(overall.sources);
      } 
      else if (overall.raw_details && overall.raw_details.length > 0) {
        const allSources = { corroboration: [], fact_checks: [] };

        overall.raw_details.forEach(detail => {
          if (detail.evidence && Array.isArray(detail.evidence)) {
            allSources.corroboration.push(...detail.evidence);
          }
          if (detail.fact_check && Array.isArray(detail.fact_check.fact_checks)) {
            allSources.fact_checks.push(...detail.fact_check.fact_checks);
          }
          if (detail.corroboration && Array.isArray(detail.corroboration.evidences)) {
            allSources.corroboration.push(...detail.corroboration.evidences);
          }
        });

        if (allSources.corroboration.length || allSources.fact_checks.length) {
          displaySources(allSources);
        }
      }
      break;

    case "IMAGE_ANALYSIS_RESULT": {
      const { url, score, explanation, session_id: responseSessionId } = message.payload;

      if (responseSessionId && responseSessionId !== sessionId) {
        console.warn("Received image result for different session:", responseSessionId);
        return; 
      }
      
      const container = document.getElementById("image-results");
      if (!container) return;
      if (container.textContent.includes("No images")) container.innerHTML = "";

      if (!document.getElementById("image-analysis-header")) {
        const header = document.createElement("h3");
        header.id = "image-analysis-header";
        header.textContent = "🖼️ Image Analysis Results";
        Object.assign(header.style, {
          margin: "10px 0 14px",
          fontSize: "15px",
          fontWeight: "700",
          color: "#334155",
          letterSpacing: "-0.3px"
        });
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
        gap: "10px",
        padding: "16px",
        borderRadius: "14px",
        marginBottom: "14px",
        background: "linear-gradient(135deg, #ffffff 0%, #fafbfc 100%)",
        border: "1px solid rgba(226, 232, 240, 0.6)",
        boxShadow: "0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)",
        transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        animation: "fadeIn 0.4s ease",
        position: "relative",
        overflow: "hidden"
      });

      // Add gradient border accent
      const borderAccent = document.createElement("div");
      Object.assign(borderAccent.style, {
        position: "absolute",
        top: "0",
        left: "0",
        width: "4px",
        height: "100%",
        background: "linear-gradient(180deg, #667eea 0%, #764ba2 100%)",
        opacity: "0",
        transition: "opacity 0.3s ease"
      });
      imgEntry.appendChild(borderAccent);

      imgEntry.addEventListener("mouseenter", () => {
        imgEntry.style.transform = "translateX(4px)";
        imgEntry.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.08), 0 2px 6px rgba(0, 0, 0, 0.08)";
        imgEntry.style.borderColor = "rgba(102, 126, 234, 0.3)";
        borderAccent.style.opacity = "1";
      });

      imgEntry.addEventListener("mouseleave", () => {
        imgEntry.style.transform = "translateX(0)";
        imgEntry.style.boxShadow = "0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.06)";
        imgEntry.style.borderColor = "rgba(226, 232, 240, 0.6)";
        borderAccent.style.opacity = "0";
      });

      let color = "#b91c1c";
      let label = "AI Generated";
      let bgGradient = "linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)";
      
      if (validity >= 75) {
        color = "#15803d";
        label = "Likely Authentic";
        bgGradient = "linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)";
      } else if (validity >= 45) {
        color = "#ca8a04";
        label = "Uncertain";
        bgGradient = "linear-gradient(90deg, rgba(234, 179, 8, 0.1) 0%, rgba(202, 138, 4, 0.05) 100%)";
      }

      const contentWrapper = document.createElement("div");
      contentWrapper.style.position = "relative";
      contentWrapper.style.zIndex = "1";

      contentWrapper.innerHTML = `
        <div style="display:flex; align-items:center; gap:14px;">
          <img src="${url}" 
              style="width:100px; height:70px; border-radius:10px; object-fit:cover; border:1px solid rgba(226, 232, 240, 0.8); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);">
          <div style="flex:1;">
            <div style="font-weight:700; color:${color}; font-size:15px; margin-bottom: 6px; letter-spacing: -0.2px;">
              ${label}
            </div>
            <div style="height:8px; background: rgba(226, 232, 240, 0.5); border-radius:4px; overflow:hidden; position: relative;">
              <div style="width:${validity}%; background:${color}; height:100%; transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1); border-radius: 4px; box-shadow: 0 0 8px ${color}40;"></div>
            </div>
            <div style="margin-top: 6px; font-size: 13px; font-weight: 600; color: ${color};">
              ${validity}% Confidence
            </div>
          </div>
        </div>
        <div style="margin-top:12px; padding: 12px; background: ${bgGradient}; border-radius: 10px; border-left: 3px solid ${color};">
          <div style="font-size:13px; color:#475569; line-height: 1.6;">
            <strong style="color: #334155;">Analysis:</strong> ${explanation || "No details available."}
          </div>
        </div>
        <div style="font-size:11px; color:#94a3b8; margin-top:8px; display: flex; align-items: center; gap: 6px;">
          <span style="width: 6px; height: 6px; background: ${color}; border-radius: 50%; display: inline-block;"></span>
          Analyzed at ${timestamp}
        </div>
      `;

      imgEntry.appendChild(contentWrapper);
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
      
    case "SESSION_CANCELLED":
      console.log("Session cancelled:", message.payload);
      if (message.payload?.session_id === sessionId) {
        console.log("Current session tasks stopped successfully");
      }
      break;
  }
});

// ---------------------------
// AUTO RUN TEXT ANALYSIS
// ---------------------------
setTimeout(analyzeTextNow, 1000);

// ---------------------------
// BADGE CLICK TOGGLE
// ---------------------------
badge.onclick = () => {
  const computed = getComputedStyle(panel);
  const isVisible = computed.display !== "none" && parseFloat(computed.opacity) > 0.5;

  if (isVisible) {
    hidePanel();
  } else {
    showPanel();
  }
};

// ---------------------------
// ESC KEY CLOSE SUPPORT
// ---------------------------
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && getComputedStyle(panel).display !== "none") {
    hidePanel();
  }
});

// ---------------------------
// DRAGGABLE PANEL
// ---------------------------
let isDragging = false;
let dragOffsetX = 0;
let dragOffsetY = 0;

header.addEventListener("mousedown", (e) => {
  isDragging = true;
  const rect = panel.getBoundingClientRect();
  dragOffsetX = e.clientX - rect.left;
  dragOffsetY = e.clientY - rect.top;
  header.style.cursor = "grabbing";
  panel.style.transition = "none";
});

document.addEventListener("mousemove", (e) => {
  if (!isDragging) return;

  let left = e.clientX - dragOffsetX;
  let top = e.clientY - dragOffsetY;

  left = Math.min(Math.max(left, -panel.offsetWidth * 0.6), window.innerWidth - panel.offsetWidth * 0.4);
  top = Math.min(Math.max(top, -panel.offsetHeight * 0.6), window.innerHeight - panel.offsetHeight * 0.4);

  panel.style.left = left + "px";
  panel.style.top = top + "px";
  panel.style.right = "auto";
  panel.style.bottom = "auto";
});

document.addEventListener("mouseup", () => {
  if (!isDragging) return;
  isDragging = false;
  header.style.cursor = "grab";
  panel.style.transition = "all 0.4s cubic-bezier(0.4, 0, 0.2, 1)";
});

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

window.addEventListener('unload', () => {
  cancelSession();
});

let heartbeatInterval = setInterval(() => {
  if (document.hidden) return;

  console.log("Session active:", sessionId);
}, 60000); 

window.addEventListener('unload', () => {
  clearInterval(heartbeatInterval);
});

if (typeof chrome !== 'undefined' && chrome.runtime) {
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === 'T') {
      chrome.runtime.sendMessage({
        type: "CHECK_SESSION_TASKS",
        payload: { session_id: sessionId }
      }, (response) => {
        console.log("Active tasks for session:", response);
      });
    }
  });
}

// ---------------------------
// SESSION INFO (MODERN BADGE)
// ---------------------------
const sessionInfo = document.createElement("div");
Object.assign(sessionInfo.style, {
  fontSize: "10px",
  color: "#94a3b8",
  padding: "10px 12px",
  borderTop: "1px solid rgba(226, 232, 240, 0.8)",
  marginTop: "12px",
  fontFamily: "'SF Mono', 'Monaco', 'Courier New', monospace",
  background: "linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%)",
  borderRadius: "10px",
  display: "flex",
  alignItems: "center",
  gap: "6px",
  transition: "all 0.2s ease"
});
sessionInfo.innerHTML = `
  <span style="width: 6px; height: 6px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: inline-block; animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;"></span>
  Session: <span style="font-weight: 600; color: #667eea;">${sessionId.substring(0, 8)}...</span>
`;
sessionInfo.title = `Full Session ID: ${sessionId}`;

sessionInfo.addEventListener("mouseenter", () => {
  sessionInfo.style.background = "linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)";
  sessionInfo.style.transform = "translateX(2px)";
});

sessionInfo.addEventListener("mouseleave", () => {
  sessionInfo.style.background = "linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%)";
  sessionInfo.style.transform = "translateX(0)";
});

panel.appendChild(sessionInfo);

console.log("✨ TrustMeter initialized with modern UI");
console.log("📊 Session ID:", sessionId); 