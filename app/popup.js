document.getElementById("openOptions").addEventListener("click", () => {
  try { chrome.runtime.openOptionsPage(); } catch (e) { alert("Open options failed"); }
});

chrome.storage.local.get({ lastResult: null }, (items) => {
  if (items.lastResult) {
    document.getElementById("last").textContent = Math.round(items.lastResult.score*100) + "%";
    document.getElementById("explain").textContent = items.lastResult.explanation;
  }
});
