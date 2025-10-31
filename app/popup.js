// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Prevent popup from closing on certain events



// ==========================================
// LIVE LOG STREAMING
// ==========================================
let eventSource = null;
let analysisStartTime = null;

function startLogStream(sessionId) {
  stopLogStream(); // Clean up any existing stream
  
  analysisStartTime = Date.now();
  
  const logDisplay = $("factDisplay");
  if (!logDisplay) return;
  
  eventSource = new EventSource(`http://127.0.0.1:5000/stream_logs/${sessionId}`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      
      if (data.type === "done") {
        stopLogStream();
        return;
      }
      
      if (data.type === "heartbeat") return;
      
      // Update log display with icon based on type
      const icons = {
        info: "🔍",
        success: "✅",
        warn: "⚠️",
        error: "❌",
        phase: "▶️"
      };
      
      const icon = icons[data.type] || "•";
      logDisplay.textContent = `${icon} ${data.message}`;
      
      // Add subtle animation
      logDisplay.style.opacity = "0.7";
      setTimeout(() => {
        logDisplay.style.opacity = "1";
      }, 100);
      
    } catch (e) {
      console.error("Log stream parse error:", e);
    }
  };
  
  eventSource.onerror = (err) => {
    console.warn("Log stream disconnected " , err);
    stopLogStream();

    if (analysisStartTime) {
    const partialMsg = "Analysis may have completed partially—check results.";
    const factDisplay = $("factDisplay");
    if (factDisplay) factDisplay.textContent = partialMsg;
  }
  };
}

function stopLogStream() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function showAnalysisTime() {
  if (!analysisStartTime) return;
  
  const duration = (Date.now() - analysisStartTime) / 1000; // seconds
  const resultsDiv = $("results");
  if (!resultsDiv) return;
  
  let emoji = "⚡";
  let message = "";
  
  if (duration < 2) {
    emoji = "🚀⚡";
    message = "Lightning fast";
  } else if (duration < 5) {
    emoji = "⚡";
    message = "Quick";
  } else if (duration < 10) {
    emoji = "✓";
    message = "Fast";
  } else if (duration < 30) {
    emoji = "⏱️";
    message = "Thorough";
  } else if (duration < 90) {
    emoji = "🕐";
    message = "Comprehensive";
  } else {
    emoji = "⏳";
    message = "Deep analysis";
  }
  
  const timeBox = document.createElement("div");
  timeBox.style.cssText = `
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
  `;
  
  timeBox.innerHTML = `
    <span style="font-size: 16px;">${emoji}</span>
    <span>${message} analysis completed in ${duration.toFixed(1)}s</span>
  `;
  
  resultsDiv.insertBefore(timeBox, resultsDiv.firstChild);
  
  analysisStartTime = null;
}

function $(id) {
  const el = document.getElementById(id);
  if (!el) console.warn(`⚠️ Missing element: #${id}`);
  return el;
}

// ==========================================
// GLOBAL VARIABLES
// ==========================================
let factInterval = null;

const facts = [
  "Misinformation spreads 6x faster than truth on social media.",
  "Always verify sources before sharing.",
  "Fact-checking sites like Snopes help debunk myths.",
  "Deepfakes are AI-generated fake videos.",
  "Critical thinking helps spot fake news.",
  "Echo chambers reinforce false beliefs.",
  "Bots amplify fake news artificially.",
  "Emotional content spreads faster.",
  "Repeated lies seem more believable.",
  "Fake news mixes truth with lies."
];

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  initializeTrustMeterUI();
});

function initializeTrustMeterUI() {
  const checkTextBtn = $("checkText");
  const clickToCheckBtn = $("clickToCheck");

  if (checkTextBtn) checkTextBtn.addEventListener("click", handleTextCheck);
  if (clickToCheckBtn) clickToCheckBtn.addEventListener("click", async () =>{ await injectImageOverlays() , setTimeout(()=> window.close() , 10)});

  renderHistory();
}

// ==========================================
// ROTATING FACTS DISPLAY
// ==========================================
function startFactsRotation() {
  const factDisplay = $("factDisplay");
  if (!factDisplay) return;

  let factIndex = Math.floor(Math.random() * facts.length);
  factDisplay.innerText = facts[factIndex];

  factInterval = setInterval(() => {
    let newIndex;
    do {
      newIndex = Math.floor(Math.random() * facts.length);
    } while (newIndex === factIndex);
    factIndex = newIndex;
    factDisplay.innerText = facts[factIndex];
  }, 2500);
}

function stopFactsRotation() {
  if (factInterval) {
    clearInterval(factInterval);
    factInterval = null;
  }
  const factDisplay = $("factDisplay");
  if (factDisplay) factDisplay.innerText = "";
}

// ==========================================
// UI STATE MANAGEMENT
// ==========================================
function showLoading() {
  const loading = $("loadingContainer");
  const results = $("results");
  
  if (results) {
    results.classList.add("hidden");
  }
  if (loading) {
    loading.classList.remove("hidden");
  }
  
   const factDisplay = $("factDisplay");
  if (factDisplay) {
    factDisplay.textContent = "🔄 Initializing...";
  }
}

function hideLoading() {
  const loading = $("loadingContainer");
  if (loading) {
    loading.classList.add("hidden");
  }
  // stopFactsRotation();
    stopLogStream();
}

function showResults() {
  const results = $("results");
  if (results) {
    results.classList.remove("hidden");
  }
}

function calculateConfidenceScore(result) {
  // If score is already provided and valid, use it
  if (result.score != null && result.score > 0) {
    return result.score > 1 ? Math.round(result.score) : Math.round(result.score * 100);
  }
  
  // Calculate score based on prediction
  const prediction = (result.prediction || "Unknown").toLowerCase();
  
  if (prediction === "real") return Math.floor(Math.random() * 15) + 85; // 85-100%
  if (prediction === "fake") return Math.floor(Math.random() * 20) + 10; // 10-30%
  if (prediction === "misleading") return Math.floor(Math.random() * 20) + 40; // 40-60%
  
  return 50; // Unknown/default
}

// ==========================================
// DISPLAY RESULT
// ==========================================
function displayResult(result) {
  hideLoading();
  showResults();

  const resultsDiv = $("results");
  if (!resultsDiv) return;

  resultsDiv.innerHTML = "";

  const prediction = result.prediction || "Unknown";
  const explanation = result.explanation || "No explanation provided";
  const text = result.input_text || result.text || "unknown";

  const predictionClass = prediction.toLowerCase();

  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
    <div class="result-header">
      <span class="result-label">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Trust Analysis
      </span>
      <span class="prediction-badge ${predictionClass}">${prediction}</span>
    </div>
    <div class="result-explanation">
      <strong>Analysis:</strong> ${explanation}
    </div>
  `;

  resultsDiv.appendChild(card);

  showAnalysisTime();

  // Save to history
  saveToHistory({
    score:calculateConfidenceScore(result),
    prediction: prediction,
    explanation: explanation,
    text: text
  });

  // Show confirmation popup for fake/misleading content
  if (prediction.toLowerCase() === "fake" || prediction.toLowerCase() === "misleading") {
    setTimeout(() => {
      showConfirmationPopup(text, explanation);
    }, 20000);
  }
}

// ==========================================
// ERROR & FEEDBACK MESSAGES
// ==========================================
function displayError(message) {
  hideLoading();
  showResults();

  const resultsDiv = $("results");
  if (!resultsDiv) return;

  resultsDiv.innerHTML = `
    <div class="feedback-message error">
      ${message}
    </div>
  `;
}

function displayFeedbackMessage() {
  const resultsDiv = $("results");
  if (!resultsDiv) return;

  showResults();
  resultsDiv.innerHTML = `
    <div class="feedback-message success">
      Thank you for your feedback! It helps improve our analysis.
    </div>
  `;

  setTimeout(() => {
    resultsDiv.innerHTML = "";
    resultsDiv.classList.add("hidden");
  }, 4000);
}

// ==========================================
// TEXT ANALYSIS HANDLER
// ==========================================
async function handleTextCheck() {
  showLoading();

  try {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript(
      {
        target: { tabId: tab.id },
        func: () => window.getSelection().toString()
      },
      (results) => {
        if (chrome.runtime.lastError || !results || !results[0]) {
          displayError("Failed to extract selected text.");
          return;
        }

        const textContent = results[0].result.trim();
        if (!textContent) {
          displayError("No text selected. Please select some text first.");
          return;
        }
        const sessionId = crypto.randomUUID();

        chrome.runtime.sendMessage(
          { type: "ANALYZE_TEXT", payload: { text: textContent } },
          (response) => {
            if (!response || response.error) {
              displayError(response?.error || "Text analysis failed.");
              return;
            }

            
            if (response.error) {
              displayError(response.error);
              return;
            }

            displayResult({
              score: response.score || 0,
              explanation: response.explanation || "Text analyzed",
              prediction: response.prediction || "Unknown",
              text: textContent
            });
          }
        );

        startLogStream(sessionId);
      }
    );
  } catch (error) {
    displayError("An error occurred during text analysis.");
    console.error("Text check error:", error);
  }
}

// ==========================================
// IMAGE ANALYSIS OVERLAY
// ==========================================
async function injectImageOverlays() {
  try {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const imgs = Array.from(document.querySelectorAll("img")).filter(
          img => img.offsetWidth > 100 && img.offsetHeight > 100
        );

        function addOverlay(img) {
          const parent = img.parentElement;
          if (!parent) return;
          if (getComputedStyle(parent).position === "static") {
            parent.style.position = "relative";
          }

          const existing = parent.querySelector(".trustmeter-overlay");
          if (existing) existing.remove();

          const overlay = document.createElement("div");
          overlay.className = "trustmeter-overlay";
          overlay.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z" stroke="currentColor" stroke-width="2"/>
            </svg>
            <span>Check</span>
          `;
          overlay.style.cssText = `
            position: absolute;
            top: 8px;
            right: 8px;
            display: flex;
            align-items: center;
            gap: 4px;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(8px);
            color: white;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            z-index: 9999;
            transition: all 0.2s ease;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          `;

          overlay.addEventListener("mouseenter", () => {
            overlay.style.background = "rgba(102, 126, 234, 0.95)";
            overlay.style.transform = "scale(1.05)";
          });

          overlay.addEventListener("mouseleave", () => {
            overlay.style.background = "rgba(0, 0, 0, 0.85)";
            overlay.style.transform = "scale(1)";
          });

          overlay.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            // chrome.runtime.sendMessage({
            //   type: "ANALYZE_IMAGE",
            //   payload: { urls: [img.src] }
            // });
            window.dispatchEvent(new CustomEvent('analyze-image', {
    detail: { url: img.src }
  }));
          });

          parent.appendChild(overlay);
        }

        imgs.forEach(addOverlay);
        return imgs.length;
      }
    });
  } catch (error) {
    console.error("Image overlay error:", error);
  }
}

// ==========================================
// MESSAGE LISTENER
// ==========================================
chrome.runtime.onMessage.addListener((message) => {
  if (!message || !message.type) return;

  if (message.type === "IMAGE_ANALYSIS_RESULT") {
    const { url, score, explanation, prediction } = message.payload;
    displayResult({
      score: score || 0,
      explanation: explanation || "Image analyzed",
      prediction: prediction || "Unknown",
      text: url
    });
  }

  if (message.type === "ANALYSIS_ERROR") {
    displayError(message.payload || "Analysis failed.");
  }
});

// ==========================================
// HISTORY MANAGEMENT
// ==========================================
function saveToHistory(entry) {
  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");
  
  history.unshift({
    score: entry.score || 0,
    prediction: entry.prediction || "Unknown",
    explanation: entry.explanation || "No explanation",
    text: entry.text || "unknown",
    timestamp: new Date().toLocaleString()
  });

  const trimmed = history.slice(0, 10);
  localStorage.setItem("analysisHistory", JSON.stringify(trimmed));
  renderHistory();
}

function renderHistory() {
  const container = $("historyContainer");
  if (!container) return;

  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");
  container.innerHTML = "";

  if (history.length === 0) {
    container.innerHTML = `<p>No recent analyses yet.</p>`;
    return;
  }

  history.forEach((item) => {
    const card = document.createElement("div");
    card.className = "history-card";

    const predictionClass = item.prediction.toLowerCase();

    card.innerHTML = `
      <div class="history-header">
        <span class="prediction-badge ${predictionClass}">${item.prediction}</span>
        <span class="timestamp">${item.timestamp}</span>
      </div>
      <p class="score-line">Confidence: ${item.score}%</p>
      <div class="explanation">
        <p><strong>Analyzed Text:</strong> ${item.text.substring(0, 150)}${item.text.length > 150 ? '...' : ''}</p>
        <p><strong>Details:</strong> ${item.explanation}</p>
      </div>
    `;

    card.addEventListener("click", () => {
      const exp = card.querySelector(".explanation");
      const currentHeight = exp.style.maxHeight;
      exp.style.maxHeight = currentHeight === "0px" || currentHeight === ""
        ? exp.scrollHeight + "px"
        : "0px";
    });

    container.appendChild(card);
  });
}

// ==========================================
// CONFIRMATION POPUP
// ==========================================
const confirmationModal = document.createElement("div");
confirmationModal.id = "trustmeter-confirmation";
Object.assign(confirmationModal.style, {
  position: "fixed",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  zIndex: "2147483648",
  background: "rgba(255, 255, 255, 0.98)",
  backdropFilter: "blur(12px)",
  padding: "24px",
  borderRadius: "16px",
  boxShadow: "0 12px 40px rgba(0, 0, 0, 0.25)",
  display: "none",
  flexDirection: "column",
  alignItems: "center",
  gap: "20px",
  maxWidth: "320px",
  fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
});

const questionText = document.createElement("p");
questionText.textContent = "Does this content seem misleading to you?";
questionText.style.cssText = `
  font-size: 15px;
  font-weight: 600;
  color: #1e293b;
  text-align: center;
  line-height: 1.5;
`;

const buttonContainer = document.createElement("div");
buttonContainer.style.cssText = "display: flex; gap: 12px;";

const yesButton = createFeedbackButton("Yes, it's misleading", "#667eea");
yesButton.onclick = () => {
  submitFeedback("YES");
  confirmationModal.style.display = "none";
};

const noButton = createFeedbackButton("No, seems fine", "#64748b");
noButton.onclick = () => {
  submitFeedback("NO");
  confirmationModal.style.display = "none";
};

buttonContainer.appendChild(yesButton);
buttonContainer.appendChild(noButton);
confirmationModal.appendChild(questionText);
confirmationModal.appendChild(buttonContainer);
document.documentElement.appendChild(confirmationModal);

function createFeedbackButton(text, color) {
  const button = document.createElement("button");
  button.textContent = text;
  button.style.cssText = `
    padding: 10px 18px;
    border-radius: 10px;
    border: none;
    background: ${color};
    color: white;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
  `;
  button.onmouseover = () => {
    button.style.transform = "translateY(-2px)";
    button.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.15)";
  };
  button.onmouseout = () => {
    button.style.transform = "translateY(0)";
    button.style.boxShadow = "none";
  };
  return button;
}

function showConfirmationPopup(text, explanation) {
  window.currentText = text;
  window.currentExplanation = explanation;
  confirmationModal.style.display = "flex";
}

function submitFeedback(responseType) {
  fetch("http://localhost:5000/submit_feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "user-fingerprint": "user-device-" + Math.random().toString(36).substring(2)
    },
    body: JSON.stringify({
      text: window.currentText || "unknown",
      explanation: window.currentExplanation || "No explanation",
      response: responseType,
      sources: []
    })
  })
    .then(res => {
      if (!res.ok) throw new Error(`Feedback failed: ${res.status}`);
      return res.json();
    })
    .then(() => displayFeedbackMessage())
    .catch(err => {
      console.error("Feedback error:", err);
      displayError("Failed to submit feedback.");
    });
}
