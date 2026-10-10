// Content script for linkedin.com/games/*: shows the solver's transparent overlay over the page.
//
// The backend runs the same CV pipeline as the CLI on a screenshot of the viewport, so no DOM
// selectors are assumed. The overlay PNG has the screenshot's size and is stretched to the
// viewport. Press Escape (or navigate) to remove it.

const OVERLAY_ID = "lps-overlay";
const TOAST_ID = "lps-toast";

function removeById(id) {
  document.getElementById(id)?.remove();
}

function fixedElement(tag, id, style) {
  removeById(id);
  const element = document.createElement(tag);
  element.id = id;
  Object.assign(element.style, { position: "fixed", zIndex: "2147483647", pointerEvents: "none" }, style);
  document.body.appendChild(element);
  return element;
}

function toast(text, background) {
  const element = fixedElement("div", TOAST_ID, {
    right: "16px",
    bottom: "16px",
    padding: "10px 16px",
    borderRadius: "20px",
    background,
    color: "#fff",
    font: "600 14px system-ui, sans-serif",
  });
  element.textContent = text;
  setTimeout(() => element.remove(), 4000);
}

chrome.runtime.onMessage.addListener((message) => {
  if (message.type === "clear") {
    removeById(OVERLAY_ID);
  } else if (message.type === "overlay") {
    const image = fixedElement("img", OVERLAY_ID, { inset: "0", width: "100vw", height: "100vh" });
    image.src = `data:image/png;base64,${message.png}`;
    toast(`Solved in ${Math.round(message.latencyUs).toLocaleString()} µs — Esc to hide`, "#0a66c2");
  } else if (message.type === "error") {
    toast(`Solver error: ${message.message}`, "#b42318");
    console.error("[LinkedIn Puzzle Solvers]", message.message);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    removeById(OVERLAY_ID);
  }
});

// LinkedIn is a single-page app: drop a stale overlay when the route changes.
let lastPath = window.location.pathname;
setInterval(() => {
  if (window.location.pathname !== lastPath) {
    lastPath = window.location.pathname;
    removeById(OVERLAY_ID);
  }
}, 500);
