export function applyDevVisibility({
  documentRef = document,
  embeddedMode,
  devMode,
}) {
  documentRef.querySelectorAll("[data-dev-only]").forEach((node) => {
    node.hidden = !devMode;
  });
  const simulatorLink = documentRef.querySelector(
    'a[href^="/simulator.html"][data-dev-only]',
  );
  if (simulatorLink && embeddedMode && devMode) {
    simulatorLink.href = "/simulator.html?embedded=1&dev=1";
  }
}

export function createDialogController({
  dialog,
  openButton,
  closeButton,
  documentRef = document,
  windowRef = window,
}) {
  let returnFocus = null;

  function open() {
    if (!dialog || dialog.open) return;
    returnFocus = documentRef.activeElement;
    dialog.showModal();
    windowRef.setTimeout(() => closeButton?.focus(), 0);
  }

  function close() {
    if (dialog?.open) dialog.close();
  }

  function restoreFocus() {
    if (returnFocus && typeof returnFocus.focus === "function") {
      returnFocus.focus();
    }
    returnFocus = null;
  }

  function register() {
    openButton?.addEventListener("click", open);
    closeButton?.addEventListener("click", close);
    if (!dialog) return;
    dialog.addEventListener("close", restoreFocus);
    dialog.addEventListener("cancel", (event) => {
      event.preventDefault();
      close();
    });
    dialog.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        close();
      }
    });
    dialog.addEventListener("click", (event) => {
      if (event.target !== dialog) return;
      const rect = dialog.getBoundingClientRect();
      const inside =
        event.clientX >= rect.left &&
        event.clientX <= rect.right &&
        event.clientY >= rect.top &&
        event.clientY <= rect.bottom;
      if (!inside) close();
    });
  }

  return { open, close, register };
}
