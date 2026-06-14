import { edgeKey, roomLabel } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function adjacencyToText(adjacency) {
  return Object.keys(adjacency || {})
    .sort()
    .map((room) => {
      const neighbors = Array.isArray(adjacency[room])
        ? [...adjacency[room]].sort()
        : [];
      return room + ": " + neighbors.join(", ");
    })
    .join("\n");
}

export function adjacencyToEdges(adjacency) {
  const edges = [];
  const seen = new Set();
  Object.keys(adjacency || {}).forEach((room) => {
    const neighbors = Array.isArray(adjacency[room]) ? adjacency[room] : [];
    neighbors.forEach((neighbor) => {
      const key = edgeKey(room, neighbor);
      if (seen.has(key)) return;
      seen.add(key);
      const pair = key.split("|");
      edges.push({ a: pair[0], b: pair[1], support: 1 });
    });
  });
  return edges;
}

export function computePositions(rooms, width, height) {
  const cx = width / 2;
  const cy = height / 2;
  const rx = Math.max(120, width * 0.36);
  const ry = Math.max(90, height * 0.33);
  const total = Math.max(1, rooms.length);
  const positions = new Map();
  rooms.forEach((room, index) => {
    const angle = (Math.PI * 2 * index) / total - Math.PI / 2;
    positions.set(room, {
      x: cx + Math.cos(angle) * rx,
      y: cy + Math.sin(angle) * ry,
    });
  });
  return positions;
}

function svgElement(documentRef, tag) {
  return documentRef.createElementNS(SVG_NS, tag);
}

export function drawMap(svg, rooms, edges, options, documentRef = document) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const positions = computePositions(rooms, 900, 520);
  const activeSet = new Set(
    Array.isArray(options.activeRooms)
      ? options.activeRooms.filter(Boolean).map(String)
      : [],
  );
  if (options.activeRoom) activeSet.add(String(options.activeRoom));
  const occupancySet = new Set(
    Array.isArray(options.occupancyRooms)
      ? options.occupancyRooms.filter(Boolean).map(String)
      : [],
  );

  const grid = svgElement(documentRef, "g");
  for (let x = 60; x <= 840; x += 60) {
    const line = svgElement(documentRef, "line");
    Object.entries({
      x1: x,
      y1: 20,
      x2: x,
      y2: 500,
      stroke: "rgba(150, 214, 236, 0.07)",
      "stroke-width": 1,
    }).forEach(([name, value]) => line.setAttribute(name, String(value)));
    grid.appendChild(line);
  }
  for (let y = 40; y <= 480; y += 60) {
    const line = svgElement(documentRef, "line");
    Object.entries({
      x1: 30,
      y1: y,
      x2: 870,
      y2: y,
      stroke: "rgba(150, 214, 236, 0.07)",
      "stroke-width": 1,
    }).forEach(([name, value]) => line.setAttribute(name, String(value)));
    grid.appendChild(line);
  }
  svg.appendChild(grid);

  edges.forEach((edge) => {
    const start = positions.get(edge.a);
    const end = positions.get(edge.b);
    if (!start || !end) return;
    const key = edgeKey(edge.a, edge.b);
    const latest = options.latestEdge && key === options.latestEdge;
    const line = svgElement(documentRef, "line");
    Object.entries({
      x1: start.x,
      y1: start.y,
      x2: end.x,
      y2: end.y,
      "stroke-linecap": "round",
    }).forEach(([name, value]) => line.setAttribute(name, String(value)));
    if (options.mode === "reference") {
      line.setAttribute("stroke", "rgba(116, 189, 220, 0.55)");
      line.setAttribute("stroke-width", "3");
    } else {
      const support = Number(edge.support || 0);
      line.setAttribute(
        "stroke",
        latest ? "#ffbd7a" : "rgba(82, 217, 177, 0.68)",
      );
      line.setAttribute("stroke-width", String(2 + Math.log1p(support) * 2));
    }
    svg.appendChild(line);

    const support = Number(edge.support || 0);
    if (options.mode !== "reference" && support > 0) {
      const text = svgElement(documentRef, "text");
      Object.entries({
        x: (start.x + end.x) / 2,
        y: (start.y + end.y) / 2 - 6,
        fill: latest ? "#ffd7a5" : "#95ddc9",
        "text-anchor": "middle",
        "font-size": 12,
        "font-family": "Segoe UI, Noto Sans, sans-serif",
      }).forEach(([name, value]) => text.setAttribute(name, String(value)));
      text.textContent = String(Math.round(support));
      svg.appendChild(text);
    }
  });

  rooms.forEach((room) => {
    const position = positions.get(room);
    if (!position) return;
    const active = activeSet.has(room);
    const primary = options.activeRoom === room;
    const node = svgElement(documentRef, "circle");
    Object.entries({
      cx: position.x,
      cy: position.y,
      r: primary ? 30 : active ? 27 : 24,
      fill: primary ? "#48d9be" : active ? "#43a9c3" : "#2c5c73",
      stroke: active ? "#f2fff9" : "#b4d9e9",
      "stroke-width": active ? 2.6 : 1.4,
    }).forEach(([name, value]) => node.setAttribute(name, String(value)));
    svg.appendChild(node);

    if (active) {
      const badge = svgElement(documentRef, "circle");
      Object.entries({
        cx: position.x + 22,
        cy: position.y - 22,
        r: 12,
        fill: occupancySet.has(room) ? "#48d9be" : "#ffbd7a",
        stroke: "#101010",
        "stroke-width": 2,
      }).forEach(([name, value]) => badge.setAttribute(name, String(value)));
      svg.appendChild(badge);
      const badgeText = svgElement(documentRef, "text");
      Object.entries({
        x: position.x + 22,
        y: position.y - 18,
        fill: "#101010",
        "text-anchor": "middle",
        "font-size": 12,
        "font-weight": 800,
        "font-family": "Segoe UI, Noto Sans, sans-serif",
      }).forEach(([name, value]) =>
        badgeText.setAttribute(name, String(value)),
      );
      badgeText.textContent = "1";
      svg.appendChild(badgeText);
    }

    const label = svgElement(documentRef, "text");
    Object.entries({
      x: position.x,
      y: position.y + 44,
      fill: "#d8edf7",
      "text-anchor": "middle",
      "font-size": 14,
      "font-family": "Segoe UI, Noto Sans, sans-serif",
    }).forEach(([name, value]) => label.setAttribute(name, String(value)));
    label.textContent = options.roomLabels?.[room] || roomLabel(room);
    svg.appendChild(label);
  });
}
