export function resolveBackendUrl(url, baseUrl = document.baseURI) {
  if (/^[a-z][a-z\d+.-]*:/i.test(url)) return url;
  return new URL(String(url).replace(/^\/+/, ""), baseUrl).toString();
}

export async function fetchJson(url, options = {}) {
  const response = await fetch(resolveBackendUrl(url), options);
  const raw = await response.text();
  let data = null;
  try {
    data = raw ? JSON.parse(raw) : null;
  } catch (_error) {
    data = null;
  }
  if (!response.ok) {
    const detail = data?.detail
      ? String(data.detail)
      : raw || `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
}

export function resolveBackendWebSocketUrl(
  path = "presencia",
  baseUrl = document.baseURI,
) {
  const url = new URL(String(path).replace(/^\/+/, ""), baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
