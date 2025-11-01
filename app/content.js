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

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (!message?.type) return;

    if (message.type === "IMAGE_ANALYSIS_RESULT") {
      const result = message.payload;
      console.log("📊 Received image analysis result:", result);

      if (result.session_id !== sessionId) {
        console.log("Ignoring result from different session");
        return;
      }

      const targetImage = document.querySelector(`img[data-analyzing-session="${result.session_id}"]`);
      if (targetImage) {
        const overlay = targetImage.closest('.trustmeter-overlay');
        if (overlay) {
          overlay.classList.remove('analyzing');
          overlay.style.pointerEvents = 'auto';
          overlay.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24">
              <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z"
              stroke="currentColor" stroke-width="2"/>
            </svg>
            <span>Check</span>
          `;
        }

        const analyzingOverlay = document.querySelector(`.analyzing-overlay[data-session="${result.session_id}"]`);
        if (analyzingOverlay) {
          analyzingOverlay.remove();
        }

        targetImage.removeAttribute('data-analyzing-session');
      }

      const updateScores = () => {

        const allResults = document.querySelectorAll('#image-results > div[id^="result-"]');
        let totalScore = 0;
        let count = 0;

        allResults.forEach(resultDiv => {
          const scoreElement = resultDiv.querySelector('[data-score]');
          if (scoreElement) {
            totalScore += parseInt(scoreElement.dataset.score);
            count++;
          }
        });

        const badge = document.getElementById("trustmeter-badge");
        if (badge) {
          const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
          badge.textContent = `Trust Score: ${averageScore}%`;
        }

        const imageScoreElement = document.querySelector('#sub-scores div:nth-child(2) div:last-child');
        if (imageScoreElement) {
          const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
          imageScoreElement.textContent = `${averageScore}%`;
        }
      };

      const imageResultsArea = document.getElementById("image-results");
      if (imageResultsArea) {
        if (imageResultsArea.textContent === "No images analyzed.") {
          imageResultsArea.innerHTML = "";
        }
            if (message.type === "IMAGE_ANALYSIS_RESULT") {
              const result = message.payload;
              console.log("📊 Received image analysis result:", result);
              if (result.session_id !== sessionId) {
                console.log("Ignoring result from different session");
                return;
              }
              const targetImage = document.querySelector(`img[data-analyzing-session="${result.session_id}"]`);
              if (targetImage) {
                const overlaySibling = targetImage.nextElementSibling;
                if (overlaySibling && overlaySibling.classList.contains('analyzing-overlay')) {
                  overlaySibling.remove();
                }
                const overlayClosest = targetImage.closest('.trustmeter-overlay');
                if (overlayClosest) {
                  overlayClosest.classList.remove('analyzing');
                  overlayClosest.style.pointerEvents = 'auto';
                  overlayClosest.innerHTML = `
                    <svg width="14" height="14" viewBox="0 0 24 24">
                      <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z"
                      stroke="currentColor" stroke-width="2"/>
                    </svg>
                    <span>Check</span>
                  `;
                }
                const analyzingOverlay = document.querySelector(`.analyzing-overlay[data-session="${result.session_id}"]`);
                if (analyzingOverlay) {
                  analyzingOverlay.remove();
                }
                targetImage.removeAttribute('data-analyzing-session');
              }
              const allResults = document.querySelectorAll('#image-results > div[id^="result-"]');
              let totalScore = 0;
              let count = 0;
              allResults.forEach(resultDiv => {
                const scoreElement = resultDiv.querySelector('[data-score]');
                if (scoreElement) {
                  totalScore += parseInt(scoreElement.dataset.score);
                  count++;
                }
              });

              const badge = document.getElementById("trustmeter-badge");
              if (badge) {
                const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
                badge.textContent = `Trust Score: ${averageScore}%`;
              }
              const imageScoreSpan = document.querySelector('#sub-scores div:nth-child(2) div:last-child');
              if (imageScoreSpan && imageScoreSpan.tagName === 'DIV') {
                const scoreSpan = imageScoreSpan.querySelector('span');
                if (scoreSpan) {
                  const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
                  scoreSpan.textContent = `${averageScore}%`;
                }
              } else if (imageScoreSpan) {
                const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
                imageScoreSpan.textContent = `${averageScore}%`;
              }
            }
        const allResults = document.querySelectorAll('#image-results > div[id^="result-"]');
        let totalScore = 0;
        let count = 0;

        allResults.forEach(resultDiv => {
          const scoreElement = resultDiv.querySelector('[data-score]');
          if (scoreElement) {
            totalScore += parseInt(scoreElement.dataset.score);
            count++;
          }
        });

        const badge = document.getElementById("trustmeter-badge");
        if (badge) {
          const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
          badge.textContent = `Trust Score: ${averageScore}%`;
        }

        stopWorking();

        const imageScoreElement = document.querySelector('#sub-scores div:nth-child(2) div:last-child');
        if (imageScoreElement) {
          const averageScore = count > 0 ? Math.round(totalScore / count) : result.score;
          imageScoreElement.textContent = `${averageScore}%`;
        }
      }
    }

    if (message.type === "TEXT_INITIAL_RESULT" || message.type === "TEXT_RESULT") {
      const data = message.payload;
      if (data && (data.score || data.initial_analysis?.score)) {
        const score = data.score || data.initial_analysis?.score || 0;
        const textScoreDiv = document.querySelector('#sub-scores div:nth-child(1) div:last-child');
        if (textScoreDiv && textScoreDiv.tagName === 'DIV') {
          const scoreSpan = textScoreDiv.querySelector('span');
          if (scoreSpan) {
            scoreSpan.textContent = `${score}%`;
          } else {
            textScoreDiv.textContent = `${score}%`;
          }
        } else if (textScoreDiv) {
          textScoreDiv.textContent = `${score}%`;
        }

        const textResultSpan = document.getElementById('text-result');
        if (textResultSpan) {
          textResultSpan.textContent = `Text Analysis Score: ${score}%`;
          textResultSpan.style.color = '#667eea';
          textResultSpan.style.fontWeight = '700';
        }
      }
    }

    if (message.type === "ANALYSIS_ERROR") {
      console.error("❌ Analysis error:", message.payload);
      const container = document.getElementById("image-results");
      if (container) {
        const errorDiv = document.createElement("div");
        errorDiv.style.cssText = `
          padding: 12px;
          background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
          border: 1px solid rgba(239, 68, 68, 0.2);
          border-radius: 10px;
          margin-bottom: 10px;
          font-size: 13px;
          color: #ef4444;
        `;
        errorDiv.innerHTML = `❌ ${message.payload}`;
        container.insertBefore(errorDiv, container.firstChild);
      }

      const badge = document.getElementById("trustmeter-badge");
      if (badge) badge.textContent = "Trust Score: —";
    }
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

// ---------------------------
// LISTEN FOR IMAGE OVERLAY CLICKS
// ---------------------------

const overlayStyles = document.createElement('style');
overlayStyles.textContent = `
  @keyframes pulseOverlay {
    0% { transform: scale(1); opacity: 0.9; }
    50% { transform: scale(1.05); opacity: 1; }
    100% { transform: scale(1); opacity: 0.9; }
  }
  
  @keyframes imageAnalyzing {
    0% { 
      background-position: 0% 0%;
      opacity: 0.2;
    }
    50% { opacity: 0.3; }
    100% { 
      background-position: 100% 100%;
      opacity: 0.2;
    }
  }

  .trustmeter-overlay {
    transition: all 0.2s ease !important;
  }

  .trustmeter-overlay:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
  }

  .trustmeter-overlay.analyzing {
    animation: pulseOverlay 1.5s infinite ease-in-out !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.95), rgba(118, 75, 162, 0.95)) !important;
  }

  .analyzing-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(135deg, #667eea33, #764ba233);
    background-size: 400% 400%;
    animation: imageAnalyzing 3s infinite ease-in-out;
    pointer-events: none;
    z-index: 9998;
  }
`;
document.head.appendChild(overlayStyles);

window.addEventListener('analyze-image', (event) => {
  const imageUrl = event.detail.url;
  console.log("🖼️ Image overlay clicked:", imageUrl);
  
  const img = event.detail.imageElement;
  const overlay = event.detail.overlay;
  const session_id = event.detail.session_id || sessionId;

  if (img) {
    img.setAttribute('data-analyzing-session', session_id);
  }

  if (overlay) {
    overlay.classList.add('analyzing');
    overlay.style.pointerEvents = 'none'; 

    overlay.innerHTML = `
      <div class="trustmeter-spinner" style="
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-top: 2px solid white;
        border-radius: 50%;
        width: 12px;
        height: 12px;
        animation: spin 0.8s linear infinite;
        margin-right: 6px;
      "></div>
      Analyzing...
    `;
  }

  const analyzingOverlay = document.createElement('div');
  analyzingOverlay.className = 'analyzing-overlay';
  analyzingOverlay.setAttribute('data-session', session_id);
  if (img && img.parentElement) {
    img.parentElement.appendChild(analyzingOverlay);
  }

  const badge = document.getElementById("trustmeter-badge");
  if (badge) {
    badge.innerHTML = `
      Trust Score: 
      <div class="trustmeter-spinner" style="
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-top: 2px solid #fff;
        border-radius: 50%;
        width: 14px;
        height: 14px;
        animation: spin 0.8s linear infinite;
        display: inline-block;
        margin-left: 6px;
        vertical-align: middle;
      "></div>
    `;
  }
  
  const container = document.getElementById("image-results");
  if (!container) return;

  const existingResult = document.getElementById(`result-${btoa(imageUrl).slice(0, 10)}`);
  if (existingResult) existingResult.remove();
  
  const loadingDiv = document.createElement("div");
  loadingDiv.id = `loading-${btoa(imageUrl).slice(0, 10)}`;
  loadingDiv.style.cssText = `
    padding: 12px;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
  `;
  loadingDiv.innerHTML = `
    <div class="trustmeter-spinner" style="
      border: 2px solid rgba(102, 126, 234, 0.1);
      border-top: 2px solid #667eea;
      border-radius: 50%;
      width: 16px;
      height: 16px;
      animation: spin 0.8s linear infinite;
    "></div>
    <span style="color: #667eea; font-size: 13px;">Analyzing image...</span>
  `;
  container.insertBefore(loadingDiv, container.firstChild);

  chrome.runtime.sendMessage(
    { 
      type: "ANALYZE_IMAGE", 
      payload: { 
        urls: [imageUrl],
        session_id: sessionId 
      } 
    },
    (response) => {
      console.log("Image analysis response:", response);

      const loadingElement = document.getElementById(`loading-${btoa(imageUrl).slice(0, 10)}`);
      if (loadingElement) loadingElement.remove();
      
      if (response && response.error) {
        console.error("❌ Image analysis error:", response.error);
        const errorDiv = document.createElement("div");
        errorDiv.id = `result-${btoa(imageUrl).slice(0, 10)}`;
        errorDiv.style.cssText = `
          padding: 12px;
          background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
          border: 1px solid rgba(239, 68, 68, 0.2);
          border-radius: 10px;
          margin-bottom: 10px;
          font-size: 13px;
          color: #ef4444;
        `;
        errorDiv.innerHTML = `❌ Failed to analyze image: ${response.error}`;
        container.appendChild(errorDiv);

        if (badge) badge.textContent = "Trust Score: —";

        stopWorking();
        return;
      }

      if (response && response.results && response.results.length > 0) {
        const result = response.results[0]; 

        const scoreText = document.querySelector("#trustmeter-badge");
        if (scoreText) scoreText.textContent = `Trust Score: ${result.score}%`;

        const resultDiv = document.createElement("div");
        resultDiv.id = `result-${btoa(imageUrl).slice(0, 10)}`;
        resultDiv.style.cssText = `
          padding: 12px;
          background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
          border: 1px solid rgba(226, 232, 240, 0.8);
          border-radius: 10px;
          margin-bottom: 10px;
        `;

        let verdictIcon = '🤔';
        let verdictColor = '#667eea';
        if (result.score >= 70) {
          verdictIcon = '✅';
          verdictColor = '#22c55e';
        } else if (result.score <= 30) {
          verdictIcon = '❌';
          verdictColor = '#ef4444';
        }

        resultDiv.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="font-weight: 600; color: ${verdictColor};">${verdictIcon} ${result.verdict}</div>
            <div style="font-weight: 700; color: ${verdictColor};">${result.score}%</div>
          </div>
          <div style="font-size: 13px; color: #64748b; line-height: 1.5;">
            ${result.explanation}
          </div>
          <div style="margin-top: 8px; font-size: 12px; color: #94a3b8;">
            <a href="${imageUrl}" target="_blank" style="color: #667eea; text-decoration: none;">View original image →</a>
          </div>
        `;
        
        container.insertBefore(resultDiv, container.firstChild);
      }
    }
  );
});

function setWorking(msg) {
  scoreText.textContent = "Score: …";
  const tr = document.getElementById("text-result");
  if (tr) tr.textContent = msg || "Analyzing...";

  if (!badge.contains(spinner)) {
    badge.textContent = "Trust Score: …";
    badge.appendChild(spinner);
  }
}

function stopWorking() {
  if (spinner.parentElement) spinner.remove();

  const analyzingOverlays = document.querySelectorAll('.analyzing-overlay');
  analyzingOverlays.forEach(overlay => overlay.remove());

  const analyzingButtons = document.querySelectorAll('.trustmeter-overlay.analyzing');
  analyzingButtons.forEach(button => {
    button.classList.remove('analyzing');
    button.style.pointerEvents = 'auto';
    button.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24">
        <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z"
        stroke="currentColor" stroke-width="2"/>
      </svg>
      <span>Check</span>
    `;
  });

  const analyzingImages = document.querySelectorAll('img[data-analyzing-session]');
  analyzingImages.forEach(img => img.removeAttribute('data-analyzing-session'));
}

// ---------------------------
// BADGE
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
// PANEL 
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
title.innerHTML = `
  <div style="display: flex; align-items: center; gap: 8px;">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
      <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z" 
        stroke="url(#gradient)" stroke-width="2"/>
      <defs>
        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" style="stop-color: #667eea"/>
          <stop offset="100%" style="stop-color: #764ba2"/>
        </linearGradient>
      </defs>
    </svg>
    TrustMeter
  </div>
`;
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

// ---------------------------
// SECTIONS (MODERN DESIGN)
// ---------------------------
const textSection = document.createElement("div");
textSection.innerHTML = `
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="2">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
      <polyline points="14 2 14 8 20 8"></polyline>
      <line x1="16" y1="13" x2="8" y2="13"></line>
      <line x1="16" y1="17" x2="8" y2="17"></line>
    </svg>
    <strong style="color: #334155; font-size: 14px;">Text Analysis</strong>
  </div>
  <span id="text-result" style="color: #64748b; font-size: 13px; line-height: 1.6;">No analysis yet.</span>
`;
Object.assign(textSection.style, {
  fontSize: "13px",
  color: "#334155",
  lineHeight: "1.6",
  padding: "16px",
  background: "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)",
  borderRadius: "14px",
  border: "1px solid rgba(226, 232, 240, 0.8)",
  marginBottom: "16px"
});

const imageSection = document.createElement("div");
imageSection.innerHTML = `
  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="2">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
      <circle cx="8.5" cy="8.5" r="1.5"></circle>
      <polyline points="21 15 16 10 5 21"></polyline>
    </svg>
    <strong style="color: #334155; font-size: 14px;">Image Analysis</strong>
  </div>
  <div id="image-results" style="color: #64748b; font-size: 13px; line-height: 1.6;">No images analyzed.</div>
`;
Object.assign(imageSection.style, {
  fontSize: "13px",
  color: "#334155",
  lineHeight: "1.6",
  padding: "16px",
  background: "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.9) 100%)",
  borderRadius: "14px",
  border: "1px solid rgba(226, 232, 240, 0.8)"
});

const sourcesSection = document.createElement("div");
// ---------------------------
// SOURCES SECTION 
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
  display: "none"
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
panel.appendChild(scoreRow);
panel.appendChild(textSection);
panel.appendChild(sourcesSection);
panel.appendChild(imageSection);

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

function displaySources(sources) {
  const sourcesContent = document.getElementById("sources-content");
  const sourcesSection = document.getElementById("sources-section");
  
  if (!sources || (!sources.corroboration?.length && !sources.fact_checks?.length)) {
    sourcesSection.style.display = "none";
    return;
  }

  sourcesContent.innerHTML = ""; 
  sourcesSection.style.display = "block";

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

      const score = source.evidence_score || source.similarity || 0;
      const scorePercent = Math.round(score * 100);
      let scoreColor = "#15803d";
      if (scorePercent < 75) scoreColor = "#ca8a04";
      if (scorePercent < 45) scoreColor = "#b91c1c";

      let domainLabel = source.domain || "";
      try {
        if (!domainLabel && source.link) domainLabel = new URL(source.link).hostname;
      } catch (e) {
        domainLabel = source.domain || "source";
      }

      sourceCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
          <div style="flex: 1; min-width: 0;">
            <div style="font-weight: 600; font-size: 13px; color: #667eea; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
              ${source.title || "Untitled"}
            </div>
            <div style="font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
              <span style="display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; background: rgba(102, 126, 234, 0.1); border-radius: 6px; font-weight: 600;">
                🌐 ${domainLabel}
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
          ${source.snippet ? (source.snippet.length > 150 ? source.snippet.substring(0,150) + '...' : source.snippet) : 'No preview available'}
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
      factCheckCard.href = fc.url || fc.link || "#";
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

// ---------------------------
// UI STATE HELPERS
// ---------------------------
function setError(err) {
  scoreText.textContent = "Score: —";
  scoreText.style.background = "linear-gradient(135deg, rgba(254, 202, 202, 0.3) 0%, rgba(252, 165, 165, 0.2) 100%)";
  scoreText.style.border = "1px solid rgba(239, 68, 68, 0.2)";
  scoreText.style.color = "#b91c1c";
  
  const tr = document.getElementById("text-result");
  if (tr) tr.textContent = "Error: " + (err || "Analysis failed or not supported.");
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

  const tr = document.getElementById("text-result");
  if (tr) tr.textContent = explanation || "No explanation returned.";

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
  if (!textScoreCard) return;
  const scoreSpan = textScoreCard.querySelector("span");
  if (!scoreSpan) return;
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
  if (!imageScoreCard) return;
  const scoreSpan = imageScoreCard.querySelector("span");
  if (!scoreSpan) return;
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
        if (!el) return NodeFilter.FILTER_REJECT;
        try {
          if (el.closest(removeSelectors.join(","))) return NodeFilter.FILTER_REJECT;
        } catch (e) {
        }
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
        img.src && img.src.startsWith && img.src.startsWith("http") &&
        img.offsetWidth > 10 &&
        img.offsetHeight > 10
    )
    .slice(0, maxCount)
    .map(img => img.src);
}

// ---------------------------
// ANALYZE FUNCTIONS 
// ---------------------------
function analyzeTextInitial(visibleText) {
  chrome.runtime.sendMessage(
    {
      type: "ANALYZE_TEXT_INITIAL",
      payload: {
        text: visibleText,
        url: location.href,
        session_id: sessionId,
      },
    },
    (response) => {
      if (!response || response.error) {
        console.warn("Initial analysis failed:", response?.error);
        return;
      }

      const tr = document.getElementById("text-result");
      if (tr) tr.textContent = response.initial_analysis || "Gathering initial impression...";

      badge.textContent = "Analyzing…";
    }
  );
}

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

  analyzeTextInitial(visibleText);

  chrome.runtime.sendMessage(
    {
      type: "ANALYZE_TEXT",
      payload: {
        text: visibleText,
        url: location.href,
        session_id: sessionId,
      },
    },
    (response) => {
      stopWorking();
      if (!response || response.error) {
        setError(response?.error || "No response from backend.");
        console.log("Text analysis failed:", response);
        return;
      }
      setResult(response.score ?? 0, response.explanation);

      if (response.sources) {
        displaySources(response.sources);
      } else if (response.raw_details && response.raw_details.length > 0) {
        const allSources = { corroboration: [], fact_checks: [] };

        response.raw_details.forEach(detail => {
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
    }
  );
}

function analyzeImagesNow() {
  const container = document.getElementById("image-results");
  setWorking("Analyzing images...");

  const visibleImages = collectVisibleImages(3);
  if (visibleImages.length === 0) {
    stopWorking();
    if (container) container.textContent = "No images found on this page.";
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
      const container2 = document.getElementById("image-results");
      if (!response || response.error) {
        if (container2) container2.textContent = "Image analysis failed or not supported.";
      } else {
        if (container2) container2.textContent = "";
      }
      if (response && response.session_id) {
        console.log("Image analysis completed for session:", response.session_id);
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
    case "TEXT_INITIAL_RESULT":
      {
        const tr = document.getElementById("text-result");
        if (tr) tr.textContent = message.payload.initial_analysis || "Reviewing information...";
      }
      break;

    case "TEXT_ANALYSIS_RESULT":
      {
        console.log("Received TEXT_ANALYSIS_RESULT:", message.payload);
        const overall = message.payload || {};
        setResult(overall.score || 0, overall.explanation || "No explanation");

        const textScoreElement = document.getElementById('text-analysis-score');
        if (textScoreElement) {
          textScoreElement.textContent = `${overall.score || 0}%`;
        }

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
      }
      break;

    case "IMAGE_ANALYSIS_RESULT": {
      console.log("🎯 Received IMAGE_ANALYSIS_RESULT:", message.payload);
      
      const { 
        url, 
        image_source,
        score, 
        explanation, 
        prediction, 
        verdict, 
        session_id: responseSessionId 
      } = message.payload;

      const imageScoreElement = document.getElementById('image-analysis-score');
      if (imageScoreElement) {
        imageScoreElement.textContent = `${score || 0}%`;
      }

      const imageUrl = url || image_source;

      if (responseSessionId && responseSessionId !== sessionId) {
        console.warn("⚠️ Received image result for different session:", responseSessionId);
        return; 
      }
      
      const container = document.getElementById("image-results");
      if (!container) {
        console.error("❌ image-results container not found!");
        return;
      }
      
      console.log("✅ Processing image result:", {
        url: imageUrl,
        score,
        verdict: verdict || prediction,
        explanation: explanation ? explanation.substring(0, 100) : 'No explanation'
      });
      
      if (container.textContent.includes("No images")) container.innerHTML = "";

      if (!document.getElementById("image-analysis-header")) {
        const header = document.createElement("h3");
        header.id = "image-analysis-header";
        header.textContent = "";
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

      const resultVerdict = (verdict || prediction || "Unknown").toLowerCase();
      
      let color = "#b91c1c";
      let label = "🤖 AI Generated";
      let bgGradient = "linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)";
      
      console.log("🏷️ Result verdict:", resultVerdict);
      
      if (resultVerdict.includes("real") || resultVerdict.includes("authentic")) {
        color = "#15803d";
        label = "✅ Likely Real";
        bgGradient = "linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)";
      } else if (resultVerdict.includes("uncertain") || (validity >= 45 && validity < 75)) {
        color = "#ca8a04";
        label = "⚠️ Uncertain";
        bgGradient = "linear-gradient(90deg, rgba(234, 179, 8, 0.1) 0%, rgba(202, 138, 4, 0.05) 100%)";
      }

      const contentWrapper = document.createElement("div");
      contentWrapper.style.position = "relative";
      contentWrapper.style.zIndex = "1";

    contentWrapper.innerHTML = `
      <div style="
          display: flex;
          gap: 14px;
          align-items: center;
      ">
        <div style="
            width: 90px;
            height: 65px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(226,226,226,0.6);
            background: #f8fafc;
            flex-shrink: 0;
        ">
          <img src="${imageUrl}" 
            style="width: 100%; height: 100%; object-fit: cover;"
            onerror="this.parentElement.style.background='#f1f5f9'; this.remove()"
          />
        </div>

        <div style="flex: 1;">
          <div style="
              display: flex;
              justify-content: space-between;
              align-items: center;
          ">
            <span style="
                font-size: 15px;
                font-weight: 600;
                color: ${color};
            ">
              ${label}
            </span>

            <span style="
                font-size: 14px;
                font-weight: 700;
                color: ${color};
            ">
              ${validity}%
            </span>
          </div>

          <div style="
              position: relative;
              width: 100%;
              height: 6px;
              background: rgba(226,232,240,0.6);
              border-radius: 4px;
              margin-top: 8px;
              overflow: hidden;
          ">
            <div style="
                width: ${validity}%;
                height: 100%;
                background: ${color};
                transition: width .4s ease;
            "></div>
          </div>
        </div>
      </div>

      <div style="
          margin-top: 14px;
          padding: 12px 14px;
          background: rgba(248,250,252,.65);
          border-left: 3px solid ${color};
          border-radius: 10px;
          backdrop-filter: blur(8px);
      ">
        <div style="font-size: 13px; color: #475569; line-height: 1.6;">
          <strong style="color:#334155; font-weight:600;">Analysis Summary</strong><br>
          ${explanation || "No explanation available."}
        </div>
      </div>

      <div style="
          margin-top: 10px;
          font-size: 11px;
          color: #94a3b8;
          display: flex;
          align-items: center;
          gap: 6px;
      ">
        <span style="
            width: 6px;
            height: 6px;
            background: ${color};
            border-radius: 50%;
            display: inline-block;
        "></span>
        Analyzed at ${timestamp}
      </div>
    `;


      imgEntry.appendChild(contentWrapper);
      container.appendChild(imgEntry);

      updateImageAverage();
      updateBadge();
      showPanel();
      
      console.log("✅ Image result displayed successfully");
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
setTimeout(analyzeTextNow, 2000);

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