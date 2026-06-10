export function bindPanelDom(documentRef = document) {
  return Object.fromEntries(
    [...documentRef.querySelectorAll("[id]")].map((node) => [node.id, node]),
  );
}
