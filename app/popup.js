let factInterval = null;

function displayResult(result) {
  const resultsDiv = document.getElementById("results");
  const loading = document.getElementById("loadingContainer");

  // Hide loader, show results
  loading.classList.remove("show");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");

  // Clear previous results
  resultsDiv.innerHTML = "";

  const prediction = result.details[0]?.prediction || "Unknown";
  const explanation = result.explanation || "No explanation provided";

  // Color-coded badge for prediction
  let predictionColor = "background: #f3f4f6; color: #374151;";
  if (prediction.toLowerCase() === "real") {
    predictionColor = "background: #d1fae5; color: #065f46;";
  } else if (prediction.toLowerCase() === "fake") {
    predictionColor = "background: #fee2e2; color: #991b1b;";
  }

  const card = document.createElement("div");
  card.className = "result-card";

  card.innerHTML = `
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
      <span style="font-size: 14px; font-weight: 500; color: #6b7280;">Trust Prediction</span>
      <span style="padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; ${predictionColor}">
        ${prediction}
      </span>
    </div>

    <p style="color: #374151; font-size: 14px; line-height: 1.5; margin-bottom: 12px;">
      <strong>Reasoning:</strong> ${explanation}
    </p>


  `;

  console.log("Prediction:", prediction);
  console.log("Full result:", result);

  resultsDiv.appendChild(card);
}

function displayError(message) {
  const resultsDiv = document.getElementById("results");
  const loading = document.getElementById("loadingContainer");

  // Hide loader, show results
  loading.classList.remove("show");
  resultsDiv.classList.add("show");
  resultsDiv.classList.remove("hidden");
  resultsDiv.innerHTML = `<p style="color: #dc2626; font-weight: 600; padding: 16px; text-align: center; background: #fef2f2; border-radius: 8px; border: 1px solid #fecaca;">${message}</p>`;
}

// Common facts array
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

// ---------------------------
// ANALYZE TEXT
// ---------------------------
document.getElementById("checkText").addEventListener("click", async () => {
  const loading = document.getElementById("loadingContainer");
  const resultsDiv = document.getElementById("results");
  const factDisplay = document.getElementById("factDisplay");

  // Reset state - hide results, show loading
  resultsDiv.classList.remove("show");
  resultsDiv.classList.add("hidden");
  loading.classList.add("show");

  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    {
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    },
    (results) => {
      if (chrome.runtime.lastError || !results || !results[0]) {
        displayError("Failed to extract selected text.");
        return;
      }

      let textContent = results[0].result.trim();
      if (!textContent) {
        displayError("No text selected.");
        return;
      }

      // Clear previous results
      resultsDiv.innerHTML = "";

      // Show rotating facts
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

      // Send text to background for analysis
      chrome.runtime.sendMessage(
        { type: "ANALYZE_TEXT", payload: { text: textContent } },
        (response) => {
          // Stop facts + hide loading
          loading.classList.remove("show");
          clearInterval(factInterval);
          factInterval = null;
          factDisplay.innerText = "";

          if (!response || response.error) {
            displayError(response?.error || "Text analysis failed.");
            return;
          }

          displayResult({
            score: response.score || 0,
            explanation: response.explanation || "Text analyzed",
            details: response.details || [response],
          });
        }
      );
    }
  );
});

// ---------------------------
// CLICK-TO-CHECK IMAGES
// ---------------------------
// document.getElementById("clickToCheck").addEventListener("click", async () => {
//   const resultsDiv = document.getElementById("results");

//   // Hide previous results, but do NOT show loader
//   resultsDiv.classList.remove("show");
//   resultsDiv.classList.add("hidden");

//   // Inject overlays on page images
//   let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
//   chrome.scripting.executeScript({
//     target: { tabId: tab.id },
//     func: () => {
//       const imgs = Array.from(document.querySelectorAll("img")).filter(
//         img => img.offsetWidth > 10 && img.offsetHeight > 10
//       );

//       function addOverlay(img) {
//         const parent = img.parentElement;
//         if (!parent) return;
//         if (getComputedStyle(parent).position === "static") parent.style.position = "relative";

//         const existing = parent.querySelector(".misinfo-overlay");
//         if (existing) existing.remove();

//         const overlay = document.createElement("div");
//         overlay.innerText = "Check";
//         overlay.style.cssText = `
//           position: absolute;
//           top: 5px;
//           right: 5px;
//           background: rgba(0,0,0,0.7);
//           color: white;
//           padding: 2px 6px;
//           font-size: 11px;
//           border-radius: 4px;
//           cursor: pointer;
//           z-index: 9999;
//         `;
//         overlay.className = "misinfo-overlay";
//         parent.appendChild(overlay);

//         overlay.onclick = (e) => {
//           e.stopPropagation();
//           document.querySelectorAll(".misinfo-overlay").forEach(el => el.remove());

//           // Show loading state and rotating facts
//           const loading = document.getElementById("loadingContainer");
//           const resultsDiv = document.getElementById("results");
//           const factDisplay = document.getElementById("factDisplay");
//           resultsDiv.classList.remove("show");
//           resultsDiv.classList.add("hidden");
//           loading.classList.add("show");

//           // Show rotating facts
//           let factIndex = Math.floor(Math.random() * facts.length);
//           factDisplay.innerText = facts[factIndex];
//           factInterval = setInterval(() => {
//             let newIndex;
//             do {
//               newIndex = Math.floor(Math.random() * facts.length);
//             } while (newIndex === factIndex);
//             factIndex = newIndex;
//             factDisplay.innerText = facts[factIndex];
//           }, 2000);

//           // Send message to background and handle response
//           chrome.runtime.sendMessage(
//             { type: "ANALYZE_IMAGE", payload: { urls: [img.src] } },
//             (response) => {
//               // Stop facts and hide loading
//               if (factInterval) {
//                 clearInterval(factInterval);
//                 factInterval = null;
//               }
//               loading.classList.remove("show");
//               factDisplay.innerText = "";

//               if (response.error) {
//                 displayError(response.error || "Image analysis failed.");
//                 return;
//               }

//               // Display results
//               displayResult({
//                 score:  0,
//                 explanation:  "Image analyzed",
              
//               });
//             }
//           );
//         };
//       }

//       imgs.forEach(addOverlay);
//     }
//   });
// });

// // ---------------------------
// // GLOBAL MESSAGE LISTENER
// // ---------------------------
// chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
//   const resultsDiv = document.getElementById("results");
//   const loading = document.getElementById("loadingContainer");
//   const factDisplay = document.getElementById("factDisplay");

//   if (message.type === "ANALYSIS_ERROR") {
//     // Stop facts and hide loading
//     if (factInterval) {
//       clearInterval(factInterval);
//       factInterval = null;
//     }
//     loading.classList.remove("show");
//     factDisplay.innerText = "";

//     // Display error
//     displayError(message.payload || "Image analysis failed.");
//   }
// });

document.getElementById("clickToCheck").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript(
    {
      target: { tabId: tab.id },
      func: () => Array.from(document.querySelectorAll("img"))
        .filter(img => img.offsetWidth > 10 && img.offsetHeight > 10)
        .map(img => img.src),
    },
    (results) => {
      if (!results || !results[0] || results[0].result.length === 0) {
        displayError("No images found.");
        return;
      }

      const imageUrls = results[0].result;
      chrome.runtime.sendMessage(
        { type: "ANALYZE_IMAGE", payload: { urls: imageUrls } },
        (response) => {
          if (!response || response.error) {
            displayError(response?.error || "Image analysis failed.");
            return;
          }

          response.forEach((imgRes) => {
            const wrapped = {
              score: imgRes.score || 0,
              explanation: imgRes.explanation || "Image analyzed.",
              details: [imgRes],
            };
            displayResult(wrapped);
          });
        }
      );
    }
  );
});

// ---------------------------
// CLICK-TO-CHECK IMAGES
// ---------------------------
document.getElementById("clickToCheck").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => {
      function addOverlay(img) {
        const parent = img.parentElement;
        if (!parent) return;

        if (getComputedStyle(parent).position === "static") {
          parent.style.position = "relative";
        }

        const existing = parent.querySelector(".misinfo-overlay");
        if (existing) existing.remove();

        const overlay = document.createElement("div");
        overlay.innerText = "Check";
        overlay.style.position = "absolute";
        overlay.style.top = "5px";
        overlay.style.right = "5px";
        overlay.style.background = "rgba(0,0,0,0.7)";
        overlay.style.color = "white";
        overlay.style.padding = "2px 6px";
        overlay.style.fontSize = "11px";
        overlay.style.borderRadius = "4px";
        overlay.style.cursor = "pointer";
        overlay.style.zIndex = "9999";
        overlay.className = "misinfo-overlay";

        parent.appendChild(overlay);

        overlay.onclick = (e) => {
          e.stopPropagation();
          document.querySelectorAll(".misinfo-overlay").forEach(el => el.remove());

          chrome.runtime.sendMessage(
            { type: "ANALYZE_IMAGE", payload: { urls: [img.src] } }
          );
        };
      }

      const imgs = Array.from(document.querySelectorAll("img")).filter(
        img => img.offsetWidth > 10 && img.offsetHeight > 10
      );
      imgs.forEach(addOverlay);
    },
  });
});

// ---------------------------
// RECEIVE IMAGE ANALYSIS RESULTS AND UPDATE POPUP
// ---------------------------
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || !message.type) return;

  if (message.type === "IMAGE_ANALYSIS_RESULT") {
    const { url, score, explanation } = message.payload;

    const wrapped = {
      score: score || 0,
      explanation: explanation || "Image analyzed",
      details: [message.payload],
    };
    displayResult(wrapped);
  }

  if (message.type === "ANALYSIS_ERROR") {
    displayError(message.payload);
  }
}); 