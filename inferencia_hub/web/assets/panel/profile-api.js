import { fetchJson } from "./api.js";

export function fetchProfilesApi() {
  return fetchJson("/api/profiles", { cache: "no-store" });
}

export function createProfileApi(payload) {
  return fetchJson("/api/profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateProfileApi(profileId, payload) {
  return fetchJson(`/api/profiles/${profileId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function activateProfileApi(profileId) {
  return fetchJson(`/api/profiles/${profileId}/activate`, {
    method: "POST",
  });
}

export function deleteProfileApi(profileId) {
  return fetchJson(`/api/profiles/${profileId}`, { method: "DELETE" });
}

export function inferProfileLayoutApi(profileId, payload) {
  return fetchJson(`/api/profiles/${profileId}/infer-layout`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
