<p align="center">
  <img src="./docs/assets/Roaya_VL_logo.png" alt="Roa’ya-VL logo" width="220" />
</p>


<p align="center" style="margin-top:-8px; font-size:28px;">
  <b>رؤيا</b>
</p>

<h1 align="center">
  Roa’ya-VL-3B: Compression-First Visual Tokenization for Arabic–English VLMs
</h1>

<p align="center">
  <b>Yakoub Bazi</b><sup>1</sup> · <b>Mansour Zuair</b><sup>1</sup> · <b>Mohamad Mahmoud Al Rahhal</b><sup>2</sup><br/>
  <sup>1</sup>King Saud University · <sup>2</sup>King Saud University
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
Work in progress. No SOTA claims.

### What is Roa’ya-VL-3B?
Roa’ya-VL-3B is a bilingual Arabic–English vision–language model that studies whether **compression-first visual tokenization**
(inspired by OCR-style encoders) can generalize beyond OCR to broader VLM tasks under a strict visual token budget.

**Tokenization regimes**
- **Base:** 1024×1024 → **256** visual tokens  
- **Large:** 1280×1280 → **400** visual tokens  
- **Tiling:** up to **9×400** tokens for document-scale inputs / multi-image

### Resources
- **Project page:** https://yakoubbazi.github.io/Roaya-VL/
- **Code:** https://github.com/yakoubbazi/Roaya-VL
- **Paper:** soon
- **Models:** soon

### Roadmap
- [ ] Teaser examples (Arabic OCR / DocVQA / VQA / multi-image)
- [ ] Results table + training trajectory plot
- [ ] Reproducibility checklist (scripts + configs)
- [ ] Public release (weights + code + evaluation)

### Citation
```bibtex
@article{bazi2025roaya,
  title   = {Roa'ya-VL-3B: Compression-First Visual Tokenization for Arabic--English VLMs},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
