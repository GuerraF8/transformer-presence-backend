export function toPercent(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "0.0%";
}

export function toMs(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} ms` : "n/a";
}

export function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number)
    ? new Intl.NumberFormat("es-CL").format(Math.round(number))
    : "0";
}

export function trainingStateLabel(value) {
  const normalized = String(value || "idle").toLowerCase();
  if (normalized === "running") return "Entrenando";
  if (normalized === "completed") return "Completado";
  if (normalized === "error") return "Error";
  return "En espera";
}

export function roomLabel(room) {
  return String(room || "-").replace(/_/g, " ");
}

export function edgeKey(a, b) {
  return [String(a || ""), String(b || "")].sort().join("|");
}

export function formatTime(iso) {
  if (!iso) return "-";
  const date = new Date(iso);
  if (!Number.isFinite(date.getTime())) return String(iso);
  return date.toLocaleString("es-CL", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  return `${(bytes / Math.pow(1024, index)).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function isoToLocalInput(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (!Number.isFinite(date.getTime())) return "";
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 16);
}

export function localInputToIso(value) {
  if (!value) return "";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString() : "";
}
