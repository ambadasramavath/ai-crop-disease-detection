document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("leaf_image");
  const preview = document.getElementById("preview");
  const label = document.getElementById("file-label");
  const zone = document.getElementById("drop-zone");
  const form = document.getElementById("upload-form");
  if (!input || !preview || !label || !zone) return;

  function showFile(file) {
    if (!file) return;
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      alert("Please select a JPG, PNG, or WEBP image.");
      input.value = "";
      preview.hidden = true;
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      alert("The image must be 8 MB or smaller.");
      input.value = "";
      preview.hidden = true;
      return;
    }
    label.textContent = file.name;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
  }
  input.addEventListener("change", () => showFile(input.files[0]));
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragging");
    if (e.dataTransfer.files.length) {
      try { input.files = e.dataTransfer.files; } catch (_) {}
      showFile(e.dataTransfer.files[0]);
    }
  });
  if (form) form.addEventListener("submit", () => {
    const button = form.querySelector("button[type=submit]");
    if (button) { button.disabled = true; button.textContent = "Analyzing…"; }
  });
});
