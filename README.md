<p align="center">
  <img src="docs/assets/Roaya_VL_logo_w256.png" width="220" alt="Roa’ya-VL logo">
</p>

<p align="center" style="color:#4b5563; margin-top:-10px; font-size:20px; font-weight:800;">
مشروع رؤيا — الهدف: تطوير نموذج لغوي-بصري عربي–إنجليزي لفهم الصور وقراءة النصوص (OCR) والاستجابة بالاستدلال، وتقديم مرجع عملي يوضح أفضل الاستراتيجيات لبناء نماذج VLM عبر مراحل التدريب المختلفة: ما قبل التدريب، الضبط الخاضع للإشراف، وتحسين التفضيلات/التعلّم التعزيزي.</p>

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

## 🔥 Status
**Stage-2 (instruction tuning) completed up to ~50K steps.**  
- Data mixture size: **~18.0M samples** (loaded: **17,937,575**)  
- Batch size: **128**  
- Selected **best checkpoint: 45K** (used as the Stage-2 model for reporting and next stages)

---

## 📢 Latest Updates
- **2026-01-21**: Added Stage-2 validation trajectory figure (**MMBench-EN / OCRBench / TextVQA / Avg**).
- **2026-01-27**: Selected **best Stage-2 checkpoint (45K)** from the trajectory (trained up to **50K**).

---

## 🧱 Training pipeline (current: Stage 2)
<p align="center">
  <img src="docs/assets/Train_Pipeline.png" width="900" alt="Roa’ya-VL training pipeline (Stage 2)">
</p>
<p align="center"><i>Stage-2 instruction tuning: DeepSeek-OCR–inspired compression-first visual tokenization + projector + Qwen2.5-VL-3B backbone, trained on a FineVision mix with Arabic.</i></p>

---

## 📈 Stage-2 validation trajectory (reported)

<p align="center">
  <img src="docs/assets/stage2_trends_en.png" width="1100" alt="Stage-2 validation trajectory (MMBench-EN / OCRBench / TextVQA / Average)">
</p>
<p align="center"><i>Validation trend across Stage-2 checkpoints (MMBench-EN, OCRBench, TextVQA, and Average). We report the best checkpoint (~45K) as the final Stage-2 model.</i></p>

---

## What is Roa’ya-VL-3B?
We introduce **Roa’ya-VL-3B**, a bilingual **Arabic–English** vision–language model (VLM) built to evaluate whether **compression-first, OCR-style visual tokenization** can generalize beyond OCR to broader VLM tasks under a **fixed visual token budget**.

Roa’ya-VL-3B combines:
- A **DeepSeek-OCR** visual encoder (compression-first tokenization; SAM & CLIP features)
- A lightweight **projector**
- A strong open backbone **Qwen2.5-VL-3B**

The model supports token-efficient regimes (256/400 tokens) and document-scale tiling (up to 9×400 tokens). We study practical training stages (alignment → instruction tuning → Arabic consolidation → preference optimization) and provide transparent intermediate validation to understand how **data mixture**, **tokenization**, and **Arabic-focused consolidation** affect OCR fidelity, reasoning, and Arabic visual understanding.

---

## Tokenization regimes
- **Tiny:** 520×520 → **96** visual tokens  
- **Small:** 680×680 → **100** visual tokens  
- **Base:** 1024×1024 → **256** visual tokens  
- **Large:** 1280×1280 → **400** visual tokens  
- **Tiling:** up to **9×400** tokens for document-scale inputs / multi-image  

---

## Resources
- **Project page:** https://yakoubbazi.github.io/Roaya-VL/
- **Code:** https://github.com/yakoubbazi/Roaya-VL
- **Paper:** soon
- **Models:** soon

---

## Roadmap
- [x] Stage-2 instruction tuning to **~50K steps**
- [x] Pick **best checkpoint (~45K)** based on Stage-2 validation trend
- [ ] Expanded evaluation on additional benchmarks (InfoVQA, POPE, MMMU, MMStar, etc.)
- [ ] **Stage-2.5 Arabic consolidation** (Arabic instruction + OCR/doc + culture)
- [ ] Stage-3 **preference optimization** (DPO / ORPO / GRPO) when preference/reward data is ready
- [ ] Teaser examples (OCR / Doc / VQA / multi-image)
- [ ] Reproducibility checklist (scripts + configs)
- [ ] Public release (weights + code + evaluation)

---

## Acknowledgement
- **LLaVA-NeXT**: the codebase we built upon for training/evaluation utilities and core VLM engineering components.  
  https://github.com/LLaVA-VL/LLaVA-NeXT

- **DeepSeek-OCR**: the OCR-style, compression-first vision encoder inspiration we adopt/adapt for token-efficient visual tokenization in Roa’ya-VL.  
  https://github.com/deepseek-ai/DeepSeek-OCR

- **VLMEvalKit**: the evaluation toolkit we used to integrate Roa’ya-VL and run standardized evaluations across multiple VLM benchmarks.  
  https://github.com/open-compass/VLMEvalKit

> **Note:** Roa’ya-VL is an independent research project and is not affiliated with LLaVA-NeXT, DeepSeek-OCR, or VLMEvalKit.

---

## Citation
```bibtex
@article{bazi2025roaya,
  title   = {Roa'ya-VL-3B: Compression-First Visual Tokenization for Arabic-English VLMs},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}




<p align="center">
  <img src="docs/assets/Roaya_VL_logo_w256.png" width="220" alt="Roa’ya-VL logo">
</p>

<p align="center" style="color:#4b5563; margin-top:-10px; font-size:20px; font-weight:800;">
  مشروع رؤيا — إطار عمل عملي لبناء وتقييم نماذج لغوية-بصرية عربي–إنجليزي لفهم الصور وقراءة النصوص (OCR) والاستدلال.
  يهدف المشروع إلى توثيق أفضل الممارسات عبر مراحل بناء نماذج VLM: ما قبل التدريب/المحاذاة، الضبط الخاضع للإشراف (SFT)،
  ثم تحسين التفضيلات (DPO/ORPO/GRPO) أو التعلم التعزيزي عند توفر بيانات مناسبة.
</p>

<h1 align="center" style="font-size:42px; line-height:1.15; margin:10px 0 8px;">
  Roa’ya-VL: A Practical Framework for Arabic–English Vision-Language Models (OCR + Reasoning)
</h1>

<p align="center" style="color:#6b7280; font-size:16px; margin-top:-2px;">
  Reference build (current): DeepSeek-OCR vision encoder + projector + Qwen2.5-VL-3B backbone
</p>

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
    <img alt="Paper" src="https://img.shields.io/badge/Paper-in%20preparation-6c757d?style=for-the-badge">
  </a>
  <a href="https://yakoubbazi.github.io/Roaya-VL/" target="_blank">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge">
  </a>
  <a href="https://github.com/yakoubbazi/Roaya-VL" target="_blank">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-111827?style=for-the-badge">
  </a>
  <a href="#" target="_blank">
    <img alt="Models" src="https://img.shields.io/badge/Models-internal%20for%20now-f59e0b?style=for-the-badge">
  </a>
</p>

---

## 🔥 Status (as of now)
- **Stage-2 (SFT instruction tuning)** completed up to **~50K steps**
- Data mixture size: **~18.0M samples** (loaded: **17,937,575**)
- Selected **best checkpoint: ~45K** (used as the Stage-2 reference for next stages)
- **Stage-2.5 Arabic consolidation** in progress (Arabic OCR/doc + Arabic instruction + culture-oriented content)

> Note: The goal is not to lock the project to a single encoder/backbone, but to provide a **reproducible, modular recipe**
> for Arabic–English VLM development and evaluation across stages.

---

## 🎯 Project Goal
**Roa’ya-VL** is a **practical framework** for developing Arabic–English VLMs that can:
- read and reason over text inside images (**OCR / Doc / Screen / Signs**),
- answer questions grounded in images,
- improve Arabic visual understanding with targeted consolidation,
- and document best practices for **stage-wise VLM training**:
  1) alignment / pretraining (when applicable),
  2) supervised instruction tuning (SFT),
  3) preference optimization (DPO/ORPO/GRPO) or RL-style improvements.

---

## 🧩 What Roa’ya Provides
### 1) A modular architecture blueprint
- **Vision encoder (swappable)**: OCR-centric encoders or general-purpose vision towers  
- **Projector (swappable)**: MLP / Q-Former-like / lightweight adapters  
- **LLM backbone (swappable)**: strong 3B-class instruction models and beyond  
- **Training recipes** across Stage-2 / Stage-2.5 / Stage-3
- **Evaluation integration** (e.g., VLMEvalKit)

### 2) A reference implementation (current build)
Our current “reference build” uses:
- **Vision encoder:** **DeepSeek-OCR** (OCR-centric, token-efficient)
- **Backbone:** **Qwen2.5-VL-3B**
- **Projector:** **mlp2x_gelu**
- **Token regime:** fixed visual token budgets (e.g., 256/400) + optional tiling

> DeepSeek-OCR is the **current default** choice, not a requirement.
> We will report ablations with alternative vision towers to validate generality.

---

## 🧱 Training Pipeline (stage-wise)
<p align="center">
  <img src="docs/assets/Train_Pipeline.png" width="900" alt="Roa’ya-VL training pipeline">
</p>
<p align="center"><i>
Stage-wise pipeline: Base build → Stage-2 SFT (broad instruction mix) → Stage-2.5 Arabic consolidation → Stage-3 preference optimization (when ready).
Current reference build uses DeepSeek-OCR + Qwen2.5-VL-3B.
</i></p>

---

## 📈 Stage-2 Validation Trajectory (reported)
<p align="center">
  <img src="docs/assets/stage2_trends_en.png" width="1100" alt="Stage-2 validation trajectory">
</p>
<p align="center"><i>
Validation trend across Stage-2 checkpoints (MMBench-EN, OCRBench, TextVQA, Avg). Best checkpoint (~45K) is used as Stage-2 reference.
</i></p>

---

## 🏗️ Stages (How we structure development)
### Stage-2 — Supervised Instruction Tuning (SFT)
Goal: build a strong general VLM base with OCR + reasoning capability.
- broad multi-domain instruction data (with Arabic included)
- stable training + checkpoint selection via validation trajectory

### Stage-2.5 — Arabic Consolidation (targeted SFT)
Goal: increase Arabic robustness without destroying general capability.
- Arabic OCR/doc understanding
- Arabic instruction following
- culture-aware / Saudi-oriented content (optional)
- careful sampling to avoid overfitting / catastrophic forgetting

### Stage-3 — Preference Optimization (when data exists)
Goal: improve helpfulness/formatting/factuality and reduce hallucinations.
- DPO / ORPO / GRPO (depending on preference/reward data)
- Arabic preference sets are especially valuable here

---

## 🧪 Evaluation
We use **VLMEvalKit** to run standardized benchmarks and track progress across checkpoints/stages.

> We recommend evaluating frequently (e.g., every 500–1000 steps in Stage-2.5) because Arabic gains can be delayed and uneven across skills (OCR vs reasoning vs instruction following).

---

## 📦 Resources
- **Project page:** https://yakoubbazi.github.io/Roaya-VL/
- **Code:** https://github.com/yakoubbazi/Roaya-VL
- **Paper:** in preparation (will decide venue later)
- **Models:** internal for now (release plan will be announced)

---

## 🗺️ Roadmap
- [x] Stage-2 SFT to ~50K steps
- [x] Select best checkpoint (~45K)
- [ ] Expand evaluation (MMStar, MMMU, POPE, RealWorldQA, etc.)
- [ ] Stage-2.5 Arabic consolidation (ongoing)
- [ ] Stage-3 preference optimization (DPO/ORPO/GRPO) when preference data is ready
- [ ] Teaser demos (OCR / Doc / VQA / multi-image)
- [ ] Reproducibility checklist (scripts + configs)
- [ ] Public release (weights + evaluation)

---

## 🙏 Acknowledgement
- **LLaVA-NeXT** (training/evaluation utilities and engineering baseline)  
  https://github.com/LLaVA-VL/LLaVA-NeXT

- **DeepSeek-OCR** (used as the **current vision-encoder choice** and inspiration for OCR-centric token-efficient visual encoding)  
  https://github.com/deepseek-ai/DeepSeek-OCR

- **VLMEvalKit** (evaluation toolkit integration)  
  https://github.com/open-compass/VLMEvalKit

> Note: Roa’ya-VL is an independent research project and is not affiliated with the above repositories.

---

## 📌 Citation (placeholder)
```bibtex
@misc{roaya_vl_framework_2026,
  title   = {Roa'ya-VL: A Practical Framework for Arabic--English Vision-Language Models},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  year    = {2026},
  note    = {Project repository and technical report in preparation}
}

