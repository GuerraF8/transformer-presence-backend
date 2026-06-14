const SVG_NS = "http://www.w3.org/2000/svg";

export function previewLabelLines(value, maxLength = 18) {
  const words = String(value || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [""];
  const lines = [];
  for (const word of words) {
    const current = lines.at(-1) || "";
    if (!current || `${current} ${word}`.length > maxLength) {
      lines.push(word);
    } else {
      lines[lines.length - 1] = `${current} ${word}`;
    }
  }
  return lines.slice(0, 2);
}

export function renderProfilePreview({
  svg,
  profile,
  documentRef = document,
}) {
  svg.innerHTML = "";
  svg.setAttribute("viewBox", "0 0 760 360");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const rooms = profile?.rooms || [];
  if (!rooms.length) {
    const text = documentRef.createElementNS(SVG_NS, "text");
    text.setAttribute("x", "50%");
    text.setAttribute("y", "50%");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "history-chart-empty");
    text.textContent = "Añade habitaciones para construir el mapa";
    svg.appendChild(text);
    return;
  }

  const centerX = 380;
  const centerY = 160;
  const radiusX = Math.min(280, 100 + rooms.length * 25);
  const radiusY = Math.min(100, 55 + rooms.length * 8);
  const positions = new Map(
    rooms.map((room, index) => {
      const angle = (Math.PI * 2 * index) / rooms.length - Math.PI / 2;
      return [
        room.slug,
        {
          x: centerX + Math.cos(angle) * radiusX,
          y: centerY + Math.sin(angle) * radiusY,
        },
      ];
    }),
  );

  for (const edge of profile.edges || []) {
    const from = positions.get(edge[0]);
    const to = positions.get(edge[1]);
    if (!from || !to) continue;
    const line = documentRef.createElementNS(SVG_NS, "line");
    line.setAttribute("x1", from.x);
    line.setAttribute("y1", from.y);
    line.setAttribute("x2", to.x);
    line.setAttribute("y2", to.y);
    line.setAttribute("class", "profile-map-edge");
    svg.appendChild(line);
  }

  for (const room of rooms) {
    const point = positions.get(room.slug);
    const group = documentRef.createElementNS(SVG_NS, "g");
    const circle = documentRef.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", point.x);
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", 22);
    circle.setAttribute("class", "profile-map-node");
    const text = documentRef.createElementNS(SVG_NS, "text");
    text.setAttribute("x", point.x);
    text.setAttribute("y", point.y + 38);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "profile-map-label");
    const label = room.name || room.slug;
    previewLabelLines(label).forEach((line, index) => {
      const span = documentRef.createElementNS(SVG_NS, "tspan");
      span.setAttribute("x", point.x);
      span.setAttribute("dy", index === 0 ? "0" : "15");
      span.textContent = line;
      text.appendChild(span);
    });
    const title = documentRef.createElementNS(SVG_NS, "title");
    title.textContent = label;
    group.append(title, circle, text);
    svg.appendChild(group);
  }
}
