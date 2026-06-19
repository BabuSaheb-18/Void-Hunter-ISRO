// =========================
// PARTICLES
// =========================

const particles = document.getElementById("particles");

if (particles) {
  for (let i = 0; i < 60; i++) {
    const p = document.createElement("span");

    p.className = "particle";

    p.style.left = Math.random() * 100 + "vw";
    p.style.top = Math.random() * 100 + "vh";
    p.style.animationDuration = 10 + Math.random() * 20 + "s";

    particles.appendChild(p);
  }
}

// =========================
// FILE UPLOAD
// =========================

const imageInput = document.getElementById("image");
const fileName = document.getElementById("fileName");

if (imageInput && fileName) {
  imageInput.addEventListener("change", function () {
    if (this.files.length > 0) {
      fileName.innerHTML = "✔ " + this.files[0].name;
    } else {
      fileName.innerHTML = "No file selected";
    }
  });
}
