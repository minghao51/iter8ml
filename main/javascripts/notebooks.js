function toggleNotebookFullscreen(button) {
  var container = button.closest(".iframe-container");
  if (!container) {
    return;
  }
  container.classList.toggle("expanded");
  if (container.classList.contains("expanded")) {
    document.body.style.overflow = "hidden";
    button.textContent = "Exit Fullscreen";
  } else {
    document.body.style.overflow = "";
    button.textContent = "Expand";
  }
}

document.addEventListener("DOMContentLoaded", function () {
  var expandButtons = document.querySelectorAll(".notebook-expand-btn");
  expandButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      toggleNotebookFullscreen(button);
    });
  });
});
