<p align="center">
  <img src="docs/assets/Roaya_VL_logo_w256.png" width="220" alt="Roa’ya-VL logo">
</p>

<h2 align="center" style="color:#0f6d3a; font-weight:800; margin-top:-6px;">
  النموذج اللغوي البصري: رؤيا
</h2>

<h1 align="center">
  Roa’ya-VL-3B: Compression-First Visual Tokenization for Arabic–English VLMs
</h1>

<p align="center">
  <b>Yakoub Bazi</b><sup>1</sup> ·
  <b>Mansour Zuair</b><sup>1</sup> ·
  <b>Mohamad Mahmoud Al Rahhal</b><sup>2</sup>
</p>

<p align="center" style="color:#4b5563; font-size:14px; line-height:1.5; margin-top:-6px;">
  <sup>1</sup>Computer Engineering Department, College of Computer and Information Sciences, King Saud University, Riyadh 11543, Saudi Arabia<br/>
  <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
    {ybazi, zuair}@ksu.edu.sa
  </span>
  <br/><br/>
  <sup>2</sup>Applied Computer Science Department, College of Applied Computer Science, King Saud University, Riyadh 11543, Saudi Arabia<br/>
  <span style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
    malrahhal@ksu.edu.sa
  </span>
</p>

<p align="center">
  <a href="#" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-soon-6c757d?style=for-the-badge">
  </a>
  <a href="https://yakoubbazi.github.io/Roaya-VL/" target="_blank">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge">
  </a>
  <a href="https://github.com/yakoubbazi/Roaya-VL" target="_blank">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-111827?style=for-the-badge">
  </a>
  <a href="#" target="_blank">
    <img alt="Models" src="https://img.shields.io/badge/HuggingFace-soon-f59e0b?style=for-the-badge">
  </a>
</p>

---

### 🔥 Status
Work in progress.

### 📢 Latest Updates
- **2026-01-17**: Stage-2 training ongoing (~12k steps). Project page and pipeline figure added.
- **2026-01-XX**: Stage-2 checkpoint + evaluation logs (TextVQA_VAL, OCRBench, DocVQA, MMBench) — coming soon.

### What is Roa’ya-VL-3B?
We introduce Roa’ya-VL-3B, a bilingual Arabic–English VLM built from scratch to evaluate whether compression-first, OCR-style vision encoders can generalize to broader VLM tasks under a fixed visual token budget. Roa’ya-VL-3B combines a token-efficient vision encoder (DeepSeek-OCR–inspired) with Qwen2.5-VL-3B, supporting 256-token (1024×1024), 400-token (1280×1280), and tiled document settings (up to 9×400 tokens). We systematically study VLM training paradigms—pretraining, instruction tuning, and preference optimization / reinforcement learning—and report transparent intermediate validation to understand how data mixture and tokenization choices affect OCR fidelity, general reasoning, and Arabic visual understanding. The model is trained on an 18.5M open instruction mixture (including 1.5M Arabic). We will release weights, code, and configs.

**Tokenization regimes**
- **Tiny:** 520×520 → **96** visual tokens  
- **Small:** 680×680 → **100** visual tokens  
- **Base:** 1024×1024 → **256** visual tokens  
- **Large:** 1280×1280 → **400** visual tokens  
- **Tiling:** up to **9×400** tokens for document-scale inputs / multi-image

### Resources
- **Project page:** https://yakoubbazi.github.io/Roaya-VL/
- **Code:** https://github.com/yakoubbazi/Roaya-VL
- **Paper:** soon
- **Models:** soon

### Roadmap
- [ ] Teaser examples (OCR / DocVQA / VQA / multi-image)
- [ ] Results table + training trajectory plot
- [ ] Reproducibility checklist (scripts + configs)
- [ ] Public release (weights + code + evaluation)

### Training pipeline (current: Stage 2)

<p align="center">
  <img src="docs/assets/Train_piepeline.png" width="980" alt="Roa’ya-VL training pipeline (current: Stage 2)">
</p>

### Citation
```bibtex
@article{bazi2025roaya,
  title   = {Roa'ya-VL-3B: Compression-First Visual Tokenization for Arabic--English VLMs},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
