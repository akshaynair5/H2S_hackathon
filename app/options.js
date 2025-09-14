document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.sync.get({ backendUrl: "", optIn: true }, (items) => {
    document.getElementById('backendUrl').value = items.backendUrl || "";
    document.getElementById('optIn').checked = items.optIn;
  });

  document.getElementById('saveBtn').addEventListener('click', () => {
    const backendUrl = document.getElementById('backendUrl').value.trim();
    const optIn = document.getElementById('optIn').checked;
    chrome.storage.sync.set({ backendUrl, optIn }, () => {
      const s = document.getElementById('status');
      s.textContent = "Saved.";
      setTimeout(()=> s.textContent = "", 1800);
    });
  });
});
