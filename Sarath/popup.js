document.getElementById("checkImage").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.location.href
    }, async (results) => {
        const url = results[0].result;

        const response = await fetch("http://localhost:5000/detect_image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });
        const data = await response.json();

        document.getElementById("results").innerText = JSON.stringify(data, null, 2);
    });
});

document.getElementById("checkVideo").addEventListener("click", async () => {
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => window.location.href
    }, async (results) => {
        const url = results[0].result;

        const response = await fetch("http://localhost:5000/detect_video", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });
        const data = await response.json();

        document.getElementById("results").innerText = JSON.stringify(data, null, 2);
    });
});
