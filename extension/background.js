// Service worker. Clicking the toolbar action grants activeTab, which allows
// captureVisibleTab; the screenshot goes to the local backend (`python main.py serve`)
// and the transparent overlay comes back to the content script.

const BACKEND = "http://127.0.0.1:8765";
const SUPPORTED = ["queens", "zip", "tango", "crossclimb", "wend", "mini-sudoku", "patches"];

function gameFromUrl(url) {
  const match = new URL(url).pathname.match(/^\/games\/([a-z-]+)/);
  return match && SUPPORTED.includes(match[1]) ? match[1] : null;
}

async function solve(game, windowId) {
  const screenshot = await chrome.tabs.captureVisibleTab(windowId, { format: "png" });
  const response = await fetch(`${BACKEND}/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ game, image: screenshot }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `backend returned ${response.status}`);
  }
  return payload;
}

chrome.action.onClicked.addListener(async (tab) => {
  const game = tab.url ? gameFromUrl(tab.url) : null;
  if (!game) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "n/a" });
    return;
  }
  await chrome.action.setBadgeText({ tabId: tab.id, text: "…" });
  try {
    // Rejects when the tab has no content script yet (it was open before the extension loaded): reload the tab.
    await chrome.tabs.sendMessage(tab.id, { type: "clear" });
    const result = await solve(game, tab.windowId);
    await chrome.tabs.sendMessage(tab.id, { type: "overlay", png: result.overlay, latencyUs: result.profile.latency_us });
    await chrome.action.setBadgeText({ tabId: tab.id, text: "✓" });
  } catch (error) {
    await chrome.action.setBadgeText({ tabId: tab.id, text: "err" });
    await chrome.tabs.sendMessage(tab.id, { type: "error", message: String(error.message || error) }).catch(() => {});
  }
});
