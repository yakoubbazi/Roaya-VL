<p align="center">
  <img src="docs/assets/Roaya_VL_logo_w256.png" width="220" alt="Roa’ya-VL logo">
</p>

<p align="center" style="color:#4b5563; margin-top:-10px;">
  مشروع رؤيا — الهدف: بناء نموذج لغوي-بصري عربي–إنجليزي بكفاءة عالية في عدد الرموز البصرية، مع تدريب شفاف وتقييم قابل لإعادة الإنتاج.
</p>

<h1 align="center" style="font-size:42px; line-height:1.15; margin:10px 0 8px;">
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
**Stage-2 (instruction tuning) is completed (early stop).**  
Training reached **50K steps**, but was stopped due to an infrastructure/storage incident.  
We finalize Stage-2 at this point and will release the **best checkpoint** and evaluation logs.

### 📢 Latest Updates
- **2026-01-20**: Stage-2 concluded at **50K** (unplanned stop due to infrastructure/storage issue).
- **Best Stage-2 checkpoint:** **45K** — **AVG=64.04** (MMBench 69.50 / OCRBench 60.90 / TextVQA 61.72).
- **Next:** evaluate Stage-2 on additional benchmarks (DocVQA, InfoVQA, POPE, MMMU, MMStar, etc.).
- **Later:** proceed to **Stage-3 (curated refinement)** based on error-driven mining (Arabic-focused).

---

### Training pipeline (current: Stage 2)
<p align="center">
  <img src="docs/assets/Train_Piepline.png" width="900" alt="Roa’ya-VL training pipeline (Stage 2)">
</p>
<p align="center"><i>Stage-2 instruction tuning: token-efficient vision encoder + projector + Qwen2.5-VL-3B backbone.</i></p>

### 📊 Stage-2 (intermediate) validation results

**Notes:** validation results during **Stage-2 instruction tuning** (in progress).  
Scores may fluctuate across checkpoints; later stages (curation / RL) are not applied yet.

| Step | MMBench_DEV_EN | OCRBench | TextVQA_VAL | AVG (EN) |
|---:|---:|---:|---:|---:|
| 5K  | 60.69 | 49.20 | 50.33 | 53.41 |
| 10K | 59.82 | 55.30 | 55.84 | 56.99 |
| 15K | 64.85 | 56.20 | 54.78 | 58.61 |
| 20K | 63.11 | 58.80 | 54.72 | 58.88 |
| 25K | 63.54 | 56.50 | 54.96 | 58.33 |
| 30K | 67.32 | 58.50 | 60.28 | 62.03 |
| 35K | 64.67 | 58.90 | 60.23 | 61.27 |
| 40K | 64.32 | 58.10 | 62.13 | 61.52 |
| 45K | 69.50 | 60.90 | 61.72 | 64.04 |
| 48K | 65.84 | 60.70 | 61.67 | 62.74 |
| 50K | 64.15 | 62.50 | 61.83 | 62.83 |

---

### What is Roa’ya-VL-3B?
We introduce Roa’ya-VL-3B, a bilingual Arabic–English VLM built from scratch to evaluate whether compression-first, OCR-style vision encoders can generalize to broader VLM tasks under a fixed visual token budget. Roa’ya-VL-3B combines a token-efficient vision encoder (DeepSeek-OCR–inspired) with Qwen2.5-VL-3B, supporting 256-token (1024×1024), 400-token (1280×1280), and tiled document settings (up to 9×400 tokens). We systematically study VLM training paradigms—pretraining, instruction tuning, and preference optimization / reinforcement learning—and report transparent intermediate validation to understand how data mixture and tokenization choices affect OCR fidelity, general reasoning, and Arabic visual understanding. The model is trained on an 18.5M open instruction mixture (including 1.5M Arabic). We will release weights, code, and configs.

**Tokenization regimes**
- **Tiny:** 520×520 → **96** visual tokens  
- **Small:** 680×680 → **100** visual tokens  
- **Base:** 1024×1024 → **256** visual tokens  
- **Large:** 1280×1280 → **400** visual tokens  
- **Tiling:** up to **9×400** tokens for document-scale inputs / multi-image

---

### Resources
- **Project page:** https://yakoubbazi.github.io/Roaya-VL/
- **Code:** https://github.com/yakoubbazi/Roaya-VL
- **Paper:** soon
- **Models:** soon

---

### Roadmap
- [ ] Evaluate Stage-2 checkpoints on additional benchmarks (DocVQA, InfoVQA, POPE, MMMU, MMStar, etc.)
- [ ] Stage-3 curated refinement (Arabic-focused), based on error-driven mining
- [ ] Preference optimization / RL (after Stage-3)
- [ ] Teaser examples (OCR / DocVQA / VQA / multi-image)
- [ ] Results table + training trajectory plot
- [ ] Reproducibility checklist (scripts + configs)
- [ ] Public release (weights + code + evaluation)

---

### Acknowledgement

- **LLaVA-NeXT**: the codebase we built upon for training/evaluation utilities and core VLM engineering components.  
  https://github.com/LLaVA-VL/LLaVA-NeXT

- **DeepSeek-OCR**: the OCR-style, compression-first vision encoder inspiration we adopt/adapt for token-efficient visual tokenization in Roa’ya-VL.  
  https://github.com/deepseek-ai/DeepSeek-OCR

- **VLMEvalKit**: the evaluation toolkit we used to integrate Roa’ya-VL and run standardized evaluations across multiple VLM benchmarks.  
  https://github.com/open-compass/VLMEvalKit

> **Note:** Roa’ya-VL is an independent research project and is not affiliated with LLaVA-NeXT, DeepSeek-OCR, or VLMEvalKit.


### Citation
```bibtex
@article{bazi2025roaya,
  title   = {Roa'ya-VL-3B: Compression-First Visual Tokenization for Arabic-English VLMs},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}


