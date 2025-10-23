let factInterval = null;

// ---------------------------
// DUMMY DATASET FOR TESTING
// ---------------------------
const dummyResults = [
  {
    score: 0.3,
    explanation: "Text contains unreliable claims and lacks credible sources.",
    details: [{ prediction: "fake" }],
    text: "The moon is made of cheese."
  },
  {
    score: 0.45,
    explanation: "Text has some factual inaccuracies and may be misleading.",
    details: [{ prediction: "misleading" }],
    text: "Vaccines cause widespread harm without evidence."
  },
  {
    score: 0.85,
    explanation: "Text is consistent with verified information.",
    details: [{ prediction: "real" }],
    text: "The Earth orbits the Sun."
  }
];

// ---------------------------
// TEST FUNCTION
// ---------------------------
function testWithDummyData(index = 0) {
  const resultsDiv = document.getElementById("results");
  const loading = document.getElementById("loadingContainer");

  // Simulate loading
  resultsDiv.classList.remove("show");
  resultsDiv.classList.add("hidden");
  loading.classList.add("show");
  startFactsRotation();

  // Simulate API response delay
  setTimeout(() => {
    loading.classList.remove("show");
    stopFactsRotation();

    const response = dummyResults[index % dummyResults.length];
    displayResult({
      score: response.score || 0,
      explanation: response.explanation || "Text analyzed",
      details: response.details || [response],
      text: response.text || "unknown"
    });
  }, 1000);
}

// ---------------------------
// DISPLAY RESULTS
// ---------------------------
function displayResult(result) {
  const resultsDiv = document.getElementById("results");
  const loading = document.getElementById("loadingContainer");

  // Hide loader, show results
  loading.classList.remove("show");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = "";
 console.log("Result is  " , result)
  const prediction = result.details[0]?.prediction || "Unknown";
  
  console.log("Prediction " , prediction)
  const explanation = result.explanation || "No explanation provided";
  const text = result.text || "unknown";

  // Color-coded badge
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
   setTimeout(function() { showConfirmationPopup(text, explanation); }, 20000);
   
  }
  console.log("Text is ", text);
  console.log("Prediction:", prediction);
  console.log("Full result:", result);

  resultsDiv.appendChild(card);
}

// ---------------------------
// DISPLAY FEEDBACK MESSAGE
// ---------------------------
function displayFeedbackMessage() {
  const resultsDiv = document.getElementById("results");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = `
    <p style="color: #065f46; font-weight: 600; padding: 16px; text-align: center; background: #d1fae5; border-radius: 8px; border: 1px solid #a7f3d0;">
      Thank you for your feedback! It helps us improve our community-driven fact-checking.
    </p>
  `;
  // Auto-hide after 4 seconds
  setTimeout(() => {
    resultsDiv.innerHTML = "";
    resultsDiv.classList.remove("show");
    resultsDiv.classList.add("hidden");
  }, 4000);
}

// ---------------------------
// DISPLAY ERRORS
// ---------------------------
function displayError(message) {
  const resultsDiv = document.getElementById("results");
  const loading = document.getElementById("loadingContainer");

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
  const factDisplay = document.getElementById("factDisplay");
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
  if (factInterval) {
    clearInterval(factInterval);
    factInterval = null;
  }
  const factDisplay = document.getElementById("factDisplay");
  factDisplay.innerText = "";
}

// ---------------------------
// TEXT ANALYSIS
// ---------------------------
document.getElementById("checkText").addEventListener("click", async () => {
  const loading = document.getElementById("loadingContainer");
  const resultsDiv = document.getElementById("results");

  resultsDiv.classList.remove("show");
  resultsDiv.classList.add("hidden");
  loading.classList.add("show");

  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    { target: { tabId: tab.id }, func: () => window.getSelection().toString() },
    (results) => {
      if (chrome.runtime.lastError || !results || !results[0]) {
        displayError("Failed to extract selected text.");
        return;
      }

      const textContent = results[0].result.trim();
      console.log("User Selected text is " , textContent)
      if (!textContent) {
        displayError("No text selected.");
        return;
      }

      startFactsRotation();
      console.log("URL IS " , tab.url)
      
      chrome.runtime.sendMessage(
        { type: "ANALYZE_TEXT", payload: { text: textContent , url: tab.url } },
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
            details: response.details || [response],
            text: textContent
          });
        }
      );
    }
  );
});

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

        // Prevent click behind overlay
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

document.getElementById("clickToCheck").addEventListener("click", injectImageOverlays);

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
      text: url // Treat URL as text for feedback
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
questionText.textContent = "Does the text sound like fake?";
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
  console.log("User confirmed: YES, text sounds fake.");
  fetch('http://localhost:5000/submit_feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'user-fingerprint': 'user-device-' + Math.random().toString(36).substring(2)
    },
    body: JSON.stringify({
      text: window.currentText || "unknown",
      explanation: window.currentExplanation || "No explanation provided",
      response: "YES",
      sources: []
    })
  })
    .then(res => {
      if (!res.ok) throw new Error(`Feedback submission failed: ${res.status}`);
      return res.json();
    })
    .then(data => {
      console.log("Feedback response:", data);
      displayFeedbackMessage();
    })
    .catch(error => {
      console.error("Error submitting feedback:", error);
      displayError("Failed to submit feedback. Please try again.");
    });
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
noButton.onmouseout = () => { yesButton.style.transform = "translateY(0)"; };
noButton.onclick = () => {
  console.log("User confirmed: NO, text does not sound fake.");
  fetch('http://localhost:5000/submit_feedback', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'user-fingerprint': 'user-device-' + Math.random().toString(36).substring(2)
    },
    body: JSON.stringify({
      text: window.currentText || "unknown",
      explanation: window.currentExplanation || "No explanation provided",
      response: "NO",
      sources: []
    })
  })
    .then(res => {
      if (!res.ok) throw new Error(`Feedback submission failed: ${res.status}`);
      return res.json();
    })
    .then(data => {
      console.log("Feedback response:", data);
      displayFeedbackMessage();
    })
    .catch(error => {
      console.error("Error submitting feedback:", error);
      displayError("Failed to submit feedback. Please try again.");
    });
  confirmationModal.style.display = "none";
};

buttonContainer.appendChild(yesButton);
buttonContainer.appendChild(noButton);

confirmationModal.appendChild(questionText);
confirmationModal.appendChild(buttonContainer);
document.documentElement.appendChild(confirmationModal);

// Function to show the popup
function showConfirmationPopup(text, explanation) {
  window.currentText = text;
  window.currentExplanation = explanation;
  confirmationModal.style.display = "flex";
}

// Close on ESC for popup
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && confirmationModal.style.display === "flex") {
    confirmationModal.style.display = "none";
  }
});

// ---------------------------
// TRIGGER TEST (For Development)
// ---------------------------
document.getElementById("testDummyData")?.addEventListener("click", () => {
  testWithDummyData(0); // Test with "fake" prediction
});