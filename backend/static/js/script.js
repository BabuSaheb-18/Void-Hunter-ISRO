// ==========================================================
// VHNet v2 | Void Hunter | UI Controller
// ==========================================================

// 1. SPACE PARTICLES GENERATOR
const particles = document.getElementById("particles");
if (particles) {
  const PARTICLE_COUNT = 60;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const particle = document.createElement("span");
    particle.className = "particle";
    particle.style.left = `${Math.random() * 100}vw`;
    particle.style.top = `${Math.random() * 100}vh`;
    // Randomized timing for a more organic "star drift" effect
    particle.style.animationDuration = `${15 + Math.random() * 15}s`;
    particle.style.animationDelay = `${Math.random() * 10}s`;
    particles.appendChild(particle);
  }
}

// 2. FILE UPLOAD INTERFACE
const imageInput = document.getElementById("image");
const fileName = document.getElementById("fileName");

if (imageInput && fileName) {
  imageInput.addEventListener("change", function () {
    const uploadArea = document.querySelector(".upload-area");
    if (this.files && this.files.length > 0) {
      const file = this.files[0];
      fileName.innerHTML = `<strong>File Selected:</strong> ${file.name}`;
      // Visual feedback: Change border color when a file is detected
      uploadArea.style.borderColor = "var(--success)";
    } else {
      fileName.innerHTML = "No file selected";
      uploadArea.style.borderColor = "var(--primary-border)";
    }
  });
}

// 3. PERFORMANCE & SMOOTH SCROLLING
document.documentElement.style.scrollBehavior = "smooth";

// Add a "fade-in" effect to all pages on load
window.addEventListener("load", () => {
  document.body.style.opacity = "1";
  document.body.style.transition = "opacity 0.8s ease";
});
document.body.style.opacity = "0";
