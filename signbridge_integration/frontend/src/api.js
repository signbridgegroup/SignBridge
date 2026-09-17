const API_BASE = import.meta.env.VITE_SIGNBRIDGE_API_BASE || 'http://localhost:8000';

async function parseJsonOrThrow(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    // ignore, handled below
  }
  if (!response.ok) {
    const detail = body?.detail || `HTTP ${response.status}`;
    const error = new Error(detail);
    error.code = body?.code || 'request_failed';
    error.status = response.status;
    throw error;
  }
  return body;
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  return parseJsonOrThrow(response);
}

export async function fetchConfig() {
  const response = await fetch(`${API_BASE}/api/config`);
  return parseJsonOrThrow(response);
}

export async function askQuestion(question, perDomainK = 2) {
  const response = await fetch(`${API_BASE}/api/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, per_domain_k: perDomainK }),
  });
  return parseJsonOrThrow(response);
}

export async function recognizeVideo(blob, filename, mode, topK = 5) {
  const form = new FormData();
  form.append('video', blob, filename);
  form.append('mode', mode);
  form.append('top_k', String(topK));

  const response = await fetch(`${API_BASE}/api/recognize`, {
    method: 'POST',
    body: form,
  });
  return parseJsonOrThrow(response);
}

export function motionUrl(motionRelativeUrl) {
  return `${API_BASE}${motionRelativeUrl}`;
}

export async function fetchMotionsList() {
  const response = await fetch(`${API_BASE}/api/motions`);
  return parseJsonOrThrow(response);
}
