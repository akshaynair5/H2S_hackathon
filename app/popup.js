let factInterval = null;

function $(id) {
  const el = document.getElementById(id);
  if (!el) console.warn(`⚠️ Missing element: #${id}`);
  return el;
}

document.addEventListener("DOMContentLoaded", () => {
  initializeTrustMeterUI();
});

function initializeTrustMeterUI() {
  const checkTextBtn = $("checkText");
  const clickToCheckBtn = $("clickToCheck");
  const resultsDiv = $("results");
  const loadingContainer = $("loadingContainer");

  if (!resultsDiv || !loadingContainer) {
    console.error("❌ Missing essential DOM elements (results or loadingContainer).");
    return;
  }

  if (checkTextBtn) checkTextBtn.addEventListener("click", handleTextCheck);
  if (clickToCheckBtn) clickToCheckBtn.addEventListener("click", injectImageOverlays);

  renderHistory();
}

// // ---------------------------
// // TEST FUNCTION
// // ---------------------------
// function testWithDummyData(index = 0) {
//   const resultsDiv = $("results");
//   const loading = $("loadingContainer");
//   if (!resultsDiv || !loading) return;

//   resultsDiv.classList.remove("show");
//   resultsDiv.classList.add("hidden");
//   loading.classList.add("show");
//   startFactsRotation();

//   setTimeout(() => {
//     loading.classList.remove("show");
//     stopFactsRotation();

//     const response = dummyResults[index % dummyResults.length];
//     displayResult({
//       score: response.score || 0,
//       explanation: response.explanation || "Text analyzed",
//       details: response.details || [response],
//       text: response.text || "unknown"
//     });
//   }, 1000);
// }

// ---------------------------
// DISPLAY RESULTS
// ---------------------------
function displayResult(result) {
  const resultsDiv = $("results");
  const loading = $("loadingContainer");
  if (!resultsDiv || !loading) return;

  loading.classList.remove("show");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = "";

  const prediction = result.prediction || "Unknown";
  const explanation = result.explanation || "No explanation provided";
  const text = result.input_text || result.text || "unknown";

  let predictionColor = "background: #f3f4f6; color: #374151;";
  if (prediction.toLowerCase() === "real") predictionColor = "background: #d1fae5; color: #065f46;";
  else if (prediction.toLowerCase() === "fake") predictionColor = "background: #fee2e2; color: #991b1b;";
  else if (prediction.toLowerCase() === "misleading") predictionColor = "background: #feebc8; color: #9c4221;";

  const card = document.createElement("div");
  card.className = "result-card";
  card.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
      <span style="font-size: 14px; font-weight: 500; color: #6b7280;">Trust Prediction</span>
      <span style="padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; ${predictionColor}">${prediction}</span>
    </div>
    <p style="color: #374151; font-size: 14px; line-height: 1.5; margin-bottom: 12px;">
      <strong>Reasoning:</strong> ${explanation}
    </p>
  `;

  if (prediction.toLowerCase() === "fake" || prediction.toLowerCase() === "misleading") {
    setTimeout(() => { showConfirmationPopup(text, explanation); }, 6000);
  }

  console.log("Text:", text);
  console.log("Prediction:", prediction);
  console.log("Full result:", result);

  resultsDiv.appendChild(card);
}

// ---------------------------
// SCORE HANDLER
// ---------------------------
function setResult(score, explanation) {
  const percent = normalizeScore(score);
  const scoreText = $("scoreText");
  if (!scoreText) return;

  scoreText.textContent = `Score: ${percent}%`;
  if (percent >= 75) scoreText.style.color = "#0b8043";
  else if (percent >= 45) scoreText.style.color = "#e09b00";
  else scoreText.style.color = "#c42f2f";

  const textResult = $("text-result");
  if (textResult) textResult.textContent = explanation || "No explanation returned.";

  collectedScores.push(percent);
  updateBadge();
}

// ---------------------------
// SCORE HANDLER
// ---------------------------
function normalizeScore(score) {
  if (score == null) return 0;
  let normalized = score;
  if (score > 1) normalized = score; // already percentage
  else normalized = Math.round(score * 100);
  return Math.min(100, Math.max(0, normalized));
}

function getPredictionFromScore(score) {
  const val = normalizeScore(score);
  if (val >= 70) return "Real";
  if (val >= 45) return "Misleading";
  return "Fake";
}

// ---------------------------
// HISTORY STORAGE
// ---------------------------
function saveToHistory(entry) {
  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");
  history.unshift({
    score: normalizeScore(entry.score),
    prediction: entry.prediction || getPredictionFromScore(entry.score),
    explanation: entry.explanation,
    text: entry.text,
    timestamp: new Date().toLocaleString()
  });
  const trimmed = history.slice(0, 5);
  localStorage.setItem("analysisHistory", JSON.stringify(trimmed));
  renderHistory();
}

function renderHistory() {
  const container = $("historyContainer");
  if (!container) return;

  const history = JSON.parse(localStorage.getItem("analysisHistory") || "[]");
  container.innerHTML = "";

  if (history.length === 0) {
    container.innerHTML = `<p style="color:#6b7280; font-size:13px;">No history available.</p>`;
    return;
  }

  history.forEach((item) => {
    const card = document.createElement("div");
    card.className = "history-card";

    let badgeColor = "#e5e7eb", badgeText = "#374151";
    if (item.prediction === "Real") {
      badgeColor = "#d1fae5"; badgeText = "#065f46";
    } else if (item.prediction === "Fake") {
      badgeColor = "#fee2e2"; badgeText = "#991b1b";
    } else if (item.prediction === "Misleading") {
      badgeColor = "#feebc8"; badgeText = "#9c4221";
    }

    card.innerHTML = `
      <div class="history-header">
        <span class="prediction-badge" style="background:${badgeColor}; color:${badgeText};">${item.prediction}</span>
        <span class="timestamp">${item.timestamp}</span>
      </div>
      <p class="score-line">Score: ${item.score}%</p>
      <div class="explanation" style="max-height:0; overflow:hidden; transition:max-height 0.4s ease;">
        <p style="font-size:13px; color:#4b5563; margin-top:4px;">
          <strong>Reasoning:</strong> ${item.explanation}
        </p>
      </div>
    `;

    card.addEventListener("click", () => {
      const exp = card.querySelector(".explanation");
      exp.style.maxHeight = exp.style.maxHeight === "0px" || exp.style.maxHeight === ""
        ? exp.scrollHeight + "px"
        : "0px";
    });

    container.appendChild(card);
  });
}

const originalDisplayResult = displayResult;
displayResult = function (result) {
  originalDisplayResult(result);
  saveToHistory(result);
};

function displayFeedbackMessage() {
  const resultsDiv = $("results");
  if (!resultsDiv) return;

  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = `
    <p style="color: #065f46; font-weight: 600; padding: 16px; text-align: center; background: #d1fae5; border-radius: 8px; border: 1px solid #a7f3d0;">
      Thank you for your feedback! It helps us improve our community-driven fact-checking.
    </p>
  `;
  setTimeout(() => {
    resultsDiv.innerHTML = "";
    resultsDiv.classList.remove("show");
    resultsDiv.classList.add("hidden");
  }, 4000);
}

function displayError(message) {
  const resultsDiv = $("results");
  const loading = $("loadingContainer");
  if (!resultsDiv || !loading) return;

  loading.classList.remove("show");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = `<p style="color: #dc2626; font-weight: 600; padding: 16px; text-align: center; background: #fef2f2; border-radius: 8px; border: 1px solid #fecaca;">${message}</p>`;
}

// ---------------------------
// ROTATING FACTS
// ---------------------------
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
  }, 2000);
}

function stopFactsRotation() {
  const factDisplay = $("factDisplay");
  if (factInterval) {
    clearInterval(factInterval);
    factInterval = null;
  }
  if (factDisplay) factDisplay.innerText = "";
}

// ---------------------------
// TEXT ANALYSIS HANDLER
// ---------------------------
async function handleTextCheck() {
  const loading = $("loadingContainer");
  const resultsDiv = $("results");
  if (!loading || !resultsDiv) return;

  resultsDiv.classList.remove("show");
  resultsDiv.classList.add("hidden");
  loading.classList.add("show");

  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    { target: { tabId: tab.id }, func: () => window.getSelection().toString() },
    (results) => {
      if (chrome.runtime.lastError || !results || !results[0]) {
        displayError("Failed to extract selected text.");
        stopFactsRotation();
        return;
      }

      const textContent = results[0].result.trim();
      if (!textContent) {
        displayError("No text selected.");
        stopFactsRotation();
        return;
      }

      startFactsRotation();

      chrome.runtime.sendMessage(
        { type: "ANALYZE_TEXT", payload: { text: textContent } },
        (response) => {
          loading.classList.remove("show");
          stopFactsRotation();

          if (!response || response.error) {
            displayError(response?.error || "Text analysis failed.");
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
    }
  );
}

// ---------------------------
// IMAGE ANALYSIS OVERLAY
// ---------------------------
async function injectImageOverlays() {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      const imgs = Array.from(document.querySelectorAll("img")).filter(
        img => img.offsetWidth > 10 && img.offsetHeight > 10
      );

      function addOverlay(img) {
        const parent = img.parentElement;
        if (!parent) return;
        if (getComputedStyle(parent).position === "static") parent.style.position = "relative";

        const existing = parent.querySelector(".misinfo-overlay");
        if (existing) existing.remove();

        const overlay = document.createElement("div");
        overlay.innerText = "Check";
        overlay.className = "misinfo-overlay";
        overlay.style.cssText = `
          position: absolute;
          top: 5px;
          right: 5px;
          background: rgba(0,0,0,0.7);
          color: white;
          padding: 2px 6px;
          font-size: 11px;
          border-radius: 4px;
          cursor: pointer;
          z-index: 9999;
        `;
        overlay.addEventListener("click", e => e.stopPropagation());
        overlay.addEventListener("mousedown", e => e.stopPropagation());
        overlay.addEventListener("mouseup", e => e.stopPropagation());

        parent.appendChild(overlay);
        overlay.onclick = (e) => {
          e.stopPropagation();
          chrome.runtime.sendMessage({ type: "ANALYZE_IMAGE", payload: { urls: [img.src] } });
        };
      }

      imgs.forEach(addOverlay);
      return imgs.map(i => i.src);
    }
  }, (results) => {
    if (!results || !results[0] || results[0].result.length === 0) {
      displayError("No images found.");
    }
  });
}

// ---------------------------
// RECEIVE IMAGE RESULTS
// ---------------------------
chrome.runtime.onMessage.addListener((message) => {
  if (!message || !message.type) return;

  if (message.type === "IMAGE_ANALYSIS_RESULT") {
    chrome.runtime.sendMessage({ type: "EXPAND_PANEL_UI" });

    const { url, score, explanation } = message.payload;
    displayResult({
      score: score || 0,
      explanation: explanation || "Image analyzed",
      details: [message.payload],
      text: url
    });
  }

  if (message.type === "ANALYSIS_ERROR") {
    displayError(message.payload || "Analysis failed.");
  }
});

// ---------------------------
// CONFIRMATION POPUP
// ---------------------------
const confirmationModal = document.createElement("div");
confirmationModal.id = "trustmeter-confirmation";
Object.assign(confirmationModal.style, {
  position: "fixed",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  zIndex: "2147483648",
  background: "rgba(255, 255, 255, 0.95)",
  backdropFilter: "blur(10px)",
  padding: "20px",
  borderRadius: "12px",
  boxShadow: "0 8px 24px rgba(0, 0, 0, 0.2)",
  display: "none",
  flexDirection: "column",
  alignItems: "center",
  gap: "16px",
  maxWidth: "300px",
  textAlign: "center",
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  color: "#2d3748"
});
confirmationModal.setAttribute("role", "dialog");
confirmationModal.setAttribute("aria-label", "Confirmation Popup");

const questionText = document.createElement("p");
questionText.textContent = "Does selected text sound malicious to you as well?";
questionText.style.fontSize = "14px";
questionText.style.fontWeight = "600";

const buttonContainer = document.createElement("div");
Object.assign(buttonContainer.style, {
  display: "flex",
  gap: "12px"
});

const yesButton = document.createElement("button");
yesButton.textContent = "YES";
Object.assign(yesButton.style, {
  padding: "8px 16px",
  borderRadius: "8px",
  border: "none",
  background: "linear-gradient(135deg, #667eea, #764ba2)",
  color: "#ffffff",
  fontSize: "13px",
  fontWeight: "600",
  cursor: "pointer",
  transition: "transform 0.2s ease"
});
yesButton.onmouseover = () => { yesButton.style.transform = "translateY(-1px)"; };
yesButton.onmouseout = () => { yesButton.style.transform = "translateY(0)"; };
yesButton.onclick = () => {
  submitFeedback("YES");
  confirmationModal.style.display = "none";
};

const noButton = document.createElement("button");
noButton.textContent = "NO";
Object.assign(noButton.style, {
  padding: "8px 16px",
  borderRadius: "8px",
  border: "none",
  background: "linear-gradient(135deg, #667eea, #764ba2)",
  color: "#ffffff",
  fontSize: "13px",
  fontWeight: "600",
  cursor: "pointer",
  transition: "transform 0.2s ease"
});
noButton.onmouseover = () => { noButton.style.transform = "translateY(-1px)"; };
noButton.onmouseout = () => { noButton.style.transform = "translateY(0)"; };
noButton.onclick = () => {
  submitFeedback("NO");
  confirmationModal.style.display = "none";
};

buttonContainer.appendChild(yesButton);
buttonContainer.appendChild(noButton);
confirmationModal.appendChild(questionText);
confirmationModal.appendChild(buttonContainer);
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
      text: window.currentText || "unknown",
      explanation: window.currentExplanation || "No explanation provided",
      response: responseType,
      sources: []
    })
  })
    .then(res => {
      if (!res.ok) throw new Error(`Feedback submission failed: ${res.status}`);
      return res.json();
    })
    .then(() => displayFeedbackMessage())
    .catch(err => {
      console.error("Error submitting feedback:", err);
      displayError("Failed to submit feedback.");
    });
}
