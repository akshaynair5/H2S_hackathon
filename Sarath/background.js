chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkImage",
    title: "Check Fake Image",
    contexts: ["image"]
  });

  chrome.contextMenus.create({
    id: "checkVideo",
    title: "Check Fake Video",
    contexts: ["video"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkImage") {
    fetch("http://localhost:5000/detect_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: info.srcUrl })
    });
  } else if (info.menuItemId === "checkVideo") {
    fetch("http://localhost:5000/detect_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: info.srcUrl })
    });
  }
});
