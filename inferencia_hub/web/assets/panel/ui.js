export function setMiniStatus(target, text, isError) {
  target.textContent = text;
  target.className = "mini " + (isError ? "error" : "ok");
}

export function setTopStatus(target, text, isError) {
  target.textContent = text;
  target.className = "status-chip " + (isError ? "error" : "ok");
}

export function appendCell(row, text, documentRef = document) {
  const cell = documentRef.createElement("td");
  cell.textContent = text;
  row.appendChild(cell);
  return cell;
}

export function appendBadgeCell(
  row,
  text,
  tone,
  documentRef = document,
) {
  const cell = documentRef.createElement("td");
  const badge = documentRef.createElement("span");
  badge.className = "status-badge " + String(tone || "info");
  badge.textContent = text;
  cell.appendChild(badge);
  row.appendChild(cell);
  return cell;
}
