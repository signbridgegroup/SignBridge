const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';
const STORAGE_KEY = 'signbridge-color-vision';

function detectInitialMode() {
  if (!isBrowser) return 'default';
  try {
    return window.localStorage.getItem(STORAGE_KEY) === 'friendly' ? 'friendly' : 'default';
  } catch {
    return 'default';
  }
}

let currentMode = detectInitialMode();
const listeners = new Set();

export function getColorVisionMode() { return currentMode; }

export function setColorVisionMode(mode) {
  if (mode !== 'default' && mode !== 'friendly') return;
  currentMode = mode;
  if (isBrowser) {
    try { window.localStorage.setItem(STORAGE_KEY, mode); } catch {}
    document.documentElement.setAttribute('data-color-vision', mode);
  }
  listeners.forEach((listener) => listener(mode));
}

export function onColorVisionChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

if (isBrowser) document.documentElement.setAttribute('data-color-vision', currentMode);
