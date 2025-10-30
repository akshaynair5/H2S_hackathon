// ==========================================
// UTILITY FUNCTIONS
// ==========================================
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

  // restore behavior: inject overlays + send to background
  if (clickToCheckBtn)
    clickToCheckBtn.addEventListener("click", async () => {
      await injectImageOverlays();
      setTimeout(() => window.close(), 10);
    });

  renderHistory();
}

// ==========================================
// ROTATING FACTS DISPLAY
// ==========================================
function startFactsRotation() {
  const factDisplay = $("factDisplay");
  if (!factDisplay) return;

  let index = Math.floor(Math.random() * facts.length);
  factDisplay.innerText = facts[index];

  factInterval = setInterval(() => {
    let newIndex;
    do newIndex = Math.floor(Math.random() * facts.length);
    while (newIndex === index);

    index = newIndex;
    factDisplay.innerText = facts[index];
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
  $("results")?.classList.add("hidden");
  $("loadingContainer")?.classList.remove("hidden");
  startFactsRotation();
}

function hideLoading() {
  $("loadingContainer")?.classList.add("hidden");
  stopFactsRotation();
}

function showResults() {
  $("results")?.classList.remove("hidden");
}

// ==========================================
// SCORE CALCULATOR
// ==========================================
function calculateConfidenceScore(result) {
  if (result.score != null && result.score > 0) {
    return result.score > 1 ? Math.round(result.score) : Math.round(result.score * 100);
  }

  const prediction = (result.prediction || "unknown").toLowerCase();
  if (prediction === "real") return 85 + Math.floor(Math.random() * 15);
  if (prediction === "fake") return 10 + Math.floor(Math.random() * 20);
  if (prediction === "misleading") return 40 + Math.floor(Math.random() * 20);

  return 50;
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
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
          <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z"
           stroke="currentColor" stroke-width="2"/>
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

  // Save to history
  saveToHistory({
    score: calculateConfidenceScore(result),
    prediction,
    explanation,
    text
  });

  if (["fake", "misleading"].includes(prediction.toLowerCase())) {
    setTimeout(() => showConfirmationPopup(text, explanation), 20000);
  }
}

// ==========================================
// ERROR & FEEDBACK MESSAGES
// ==========================================
function displayError(message) {
  hideLoading();
  showResults();
  $("results").innerHTML = `<div class="feedback-message error">${message}</div>`;
}

function displayFeedbackMessage() {
  $("results").innerHTML = `<div class="feedback-message success">Thank you for your feedback!</div>`;
  setTimeout(() => $("results").classList.add("hidden"), 4000);
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
      results => {
        if (!results?.[0]?.result?.trim()) {
          displayError("No text selected. Please select some text first.");
          return;
        }

        const textContent = results[0].result.trim();

        chrome.runtime.sendMessage(
          { type: "ANALYZE_TEXT", payload: { text: textContent } },
          response => {
            if (!response || response.error) {
              displayError(response?.error || "Text analysis failed.");
              return;
            }

            displayResult({
              score: response.score,
              explanation: response.explanation,
              prediction: response.prediction,
              text: textContent
            });
          }
        );
      }
    );
  } catch (error) {
    console.error(error);
    displayError("An error occurred during text analysis.");
  }
}

// ==========================================
// IMAGE ANALYSIS OVERLAY (restored Option B)
// ==========================================
async function injectImageOverlays() {
  try {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const imgs = [...document.querySelectorAll("img")].filter(
          img => img.offsetWidth > 100 && img.offsetHeight > 100
        );

        imgs.forEach(img => {
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
            <svg width="14" height="14" viewBox="0 0 24 24">
              <path d="M12 2L3 7V11C3 16.55 6.84 21.74 12 23C17.16 21.74 21 16.55 21 11V7L12 2Z"
              stroke="currentColor" stroke-width="2"/>
            </svg>
            <span>Check</span>`;

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
          `;

          overlay.addEventListener("click", e => {
            e.stopPropagation();
            e.preventDefault();

            chrome.runtime.sendMessage({
              type: "ANALYZE_IMAGE",
              payload: { urls: [img.src] }
            });
          });

          parent.appendChild(overlay);
        });
      }
    });
  } catch (error) {
    console.error("Image overlay error:", error);
  }
}

// ==========================================
// MESSAGE LISTENER
// ==========================================
chrome.runtime.onMessage.addListener(message => {
  if (!message?.type) return;

  if (message.type === "IMAGE_ANALYSIS_RESULT") {
    const { url, score, explanation, prediction } = message.payload;
    displayResult({
      score,
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
// HISTORY
// ==========================================
function saveToHistory(entry) {
  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");

  history.unshift({
    score: entry.score,
    prediction: entry.prediction,
    explanation: entry.explanation,
    text: entry.text,
    timestamp: new Date().toLocaleString()
  });

  localStorage.setItem("analysisHistory", JSON.stringify(history.slice(0, 10)));
  renderHistory();
}

function renderHistory() {
  const container = $("historyContainer");
  if (!container) return;

  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");
  container.innerHTML = history.length === 0 ? `<p>No recent analyses yet.</p>` : "";

  history.forEach(item => {
    const card = document.createElement("div");
    card.className = "history-card";
    card.innerHTML = `
      <div class="history-header">
        <span class="prediction-badge ${item.prediction.toLowerCase()}">${item.prediction}</span>
        <span class="timestamp">${item.timestamp}</span>
      </div>
      <p class="score-line">Confidence: ${item.score}%</p>
      <div class="explanation">
        <p><strong>Analyzed:</strong> ${item.text.slice(0, 150)}${item.text.length > 150 ? '...' : ''}</p>
        <p><strong>Details:</strong> ${item.explanation}</p>
      </div>
    `;

    container.appendChild(card);
  });
}

// ==========================================
// CONFIRMATION POPUP + FEEDBACK
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
  boxShadow: "0 12px 40px rgba(0,0,0,0.25)",
  display: "none",
  flexDirection: "column",
  alignItems: "center",
  gap: "20px",
  maxWidth: "320px",
  fontFamily: "'Inter', sans-serif"
});

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
  return button;
}

const questionText = document.createElement("p");
questionText.textContent = "Does this content seem misleading to you?";

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

const buttonContainer = document.createElement("div");
buttonContainer.style.cssText = "display:flex; gap:12px;";
buttonContainer.append(yesButton, noButton);

confirmationModal.append(questionText, buttonContainer);
document.documentElement.appendChild(confirmationModal);

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
      text: window.currentText,
      explanation: window.currentExplanation,
      response: responseType,
      sources: []
    })
  })
    .then(res => res.ok ? res.json() : Promise.reject("Failed"))
    .then(displayFeedbackMessage)
    .catch(() => displayError("Failed to submit feedback."));
}