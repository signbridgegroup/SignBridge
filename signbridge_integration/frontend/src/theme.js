/**
 * Light/dark theme switcher. Applies a data-theme="light"|"dark" attribute
 * on <html>; every color in style.css is a CSS custom property that is
 * redefined once under [data-theme="dark"], so this module only ever
 * toggles one attribute — it never touches component markup or Three.js.
 *
 * Every browser-global touch is guarded so this module stays safely
 * importable outside a browser/DOM environment (e.g. a plain-Node test).
 */

const isBrowser = typeof window !== 'undefined' && typeof document !== 'undefined';
const STORAGE_KEY = 'signbridge-theme';

function detectInitialTheme() {
  if (!isBrowser) return 'light';
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // localStorage unavailable — fall through to system preference.
  }
  return 'light';
}

let currentTheme = detectInitialTheme();
const listeners = new Set();

export function getTheme() {
  return currentTheme;
}

export function setTheme(theme) {
  if (theme !== 'light' && theme !== 'dark') return;
  currentTheme = theme;
  if (isBrowser) {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Ignore storage failures.
    }
    document.documentElement.setAttribute('data-theme', theme);
  }
  listeners.forEach((listener) => listener(theme));
}

export function toggleTheme() {
  setTheme(currentTheme === 'light' ? 'dark' : 'light');
}

export function onThemeChange(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

if (isBrowser) {
  document.documentElement.setAttribute('data-theme', currentTheme);
}
