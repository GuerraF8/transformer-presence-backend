export function renderProfilePreview({
  svg,
  profile,
  documentRef = document,
}) {
  svg.innerHTML = "";
  const rooms = profile?.rooms || [];
  if (!rooms.length) {
    const text = documentRef.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    text.setAttribute("x", "50%");
    text.setAttribute("y", "50%");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "history-chart-empty");
    text.textContent = "Añade habitaciones para construir el mapa";
    svg.appendChild(text);
    return;
  }

  const centerX = 320;
  const centerY = 150;
  const radiusX = Math.min(230, 70 + rooms.length * 22);
  const radiusY = Math.min(105, 55 + rooms.length * 10);
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
    const line = documentRef.createElementNS(
      "http://www.w3.org/2000/svg",
      "line",
    );
    line.setAttribute("x1", from.x);
    line.setAttribute("y1", from.y);
    line.setAttribute("x2", to.x);
    line.setAttribute("y2", to.y);
    line.setAttribute("class", "profile-map-edge");
    svg.appendChild(line);
  }

  for (const room of rooms) {
    const point = positions.get(room.slug);
    const group = documentRef.createElementNS(
      "http://www.w3.org/2000/svg",
      "g",
    );
    const circle = documentRef.createElementNS(
      "http://www.w3.org/2000/svg",
      "circle",
    );
    circle.setAttribute("cx", point.x);
    circle.setAttribute("cy", point.y);
    circle.setAttribute("r", 22);
    circle.setAttribute("class", "profile-map-node");
    const text = documentRef.createElementNS(
      "http://www.w3.org/2000/svg",
      "text",
    );
    text.setAttribute("x", point.x);
    text.setAttribute("y", point.y + 38);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "profile-map-label");
    text.textContent = room.name || room.slug;
    group.append(circle, text);
    svg.appendChild(group);
  }
}
