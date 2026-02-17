<p align="center">
  <img src="docs/assets/Roaya_VL_logo_w256.png" width="220" alt="Roa’ya-VL logo">
</p>

<p align="center" style="color:#4b5563; margin-top:-10px; font-size:18px; font-weight:800; line-height:1.7;">
  مشروع رؤيا (PoC) — إطار عمل عملي لبناء وتقييم نماذج لغوية-بصرية عربي–إنجليزي لفهم الصور وقراءة النصوص داخلها (OCR) والاستدلال.
  يهدف المشروع إلى توثيق أفضل الممارسات والتجارب عبر مراحل بناء نماذج VLM: المواءمة/ما قبل التدريب (عند الحاجة)، الضبط الخاضع للإشراف (SFT)،
  ثم تحسين التفضيلات (DPO/ORPO/GRPO) أو أساليب RL عند توفر بيانات مناسبة.
</p>

<h1 align="center" style="font-size:40px; line-height:1.15; margin:10px 0 8px;">
  Roa’ya-VL: A Practical Framework for Arabic–English Vision-Language Models
</h1>

<p align="center" style="color:#6b7280; font-size:15px; margin-top:-4px; line-height:1.5;">
  Not a single model — a modular recipe for building Arabic–English VLMs (OCR + Doc + VQA + Reasoning).
  <br/>
  Current reference build: DeepSeek-OCR–style vision encoder + projector + Qwen2.5-VL-3B backbone.
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
  <a href="https://yakoubbazi.github.io/Roaya-VL/" target="_blank">
    <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge">
  </a>
  <a href="https://github.com/yakoubbazi/Roaya-VL" target="_blank">
    <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-111827?style=for-the-badge">
  </a>
  <img alt="Status" src="https://img.shields.io/badge/Status-Proof%20of%20Concept-6c757d?style=for-the-badge">
  <img alt="Models" src="https://img.shields.io/badge/Models-internal%20for%20now-f59e0b?style=for-the-badge">
</p>

<hr/>

<h2>🔥 Status (as of now)</h2>
<ul>
  <li><b>Stage-2 (SFT instruction tuning)</b> completed up to <b>~50K steps</b></li>
  <li>Data mixture size: <b>~18.0M samples</b> (loaded: <b>17,937,575</b>)</li>
  <li>Selected best checkpoint: <b>~45K</b> (used as Stage-2 reference)</li>
  <li><b>Stage-2.5 Arabic consolidation</b> is ongoing (Arabic OCR/Doc + Arabic instruction + optional culture-focused content)</li>
</ul>

<blockquote>
  <b>Important:</b> Roa’ya-VL is not meant to be locked to one encoder/backbone.
  The goal is a <b>reproducible, modular framework</b> that helps practitioners build Arabic–English VLMs and understand what changes across stages.
</blockquote>

<hr/>

<h2>🎯 Project Goal</h2>
<p>
<b>Roa’ya-VL</b> is a practical framework for developing Arabic–English VLMs that can:
</p>
<ul>
  <li>read and reason over text inside images (<b>OCR / Doc / Screen / Signs</b>),</li>
  <li>answer questions grounded in images (<b>VQA + reasoning</b>),</li>
  <li>improve Arabic visual understanding via targeted consolidation,</li>
  <li>and document best practices for <b>stage-wise VLM development</b>:</li>
</ul>
<ol>
  <li><b>Alignment / pretraining</b> (when applicable)</li>
  <li><b>Supervised instruction tuning (SFT)</b></li>
  <li><b>Preference optimization</b> (DPO/ORPO/GRPO) or RL-style improvements when preference/reward data is available</li>
</ol>

<hr/>

<h2>🧩 What Roa’ya Provides</h2>

<h3>1) Modular blueprint (swappable components)</h3>
<ul>
  <li><b>Vision encoder (swappable):</b> OCR-centric encoders or general-purpose vision towers</li>
  <li><b>Projector (swappable):</b> MLP / lightweight adapters / Q-Former-like variants</li>
  <li><b>LLM backbone (swappable):</b> 3B-class instruction backbones and beyond</li>
  <li><b>Stage recipes:</b> Stage-2 / Stage-2.5 / Stage-3 with practical guidance on sampling + stability</li>
  <li><b>Evaluation integration:</b> standardized benchmarking via toolkits (e.g., VLMEvalKit)</li>
</ul>

<h3>2) Reference build (current prototype)</h3>
<p>
The current reference build is one concrete instantiation of the framework:
</p>
<ul>
  <li><b>Vision encoder:</b> DeepSeek-OCR–style (OCR-centric, token-efficient)</li>
  <li><b>Backbone:</b> Qwen2.5-VL-3B</li>
  <li><b>Projector:</b> mlp2x_gelu</li>
  <li><b>Token regime:</b> fixed visual token budgets (e.g., 256/400) + optional tiling for document-scale inputs</li>
</ul>

<blockquote>
  <b>Note:</b> This reference build is a <b>baseline implementation</b>, not a claim that DeepSeek-OCR is the best vision tower for all settings.
  One of the project goals is to test alternative vision encoders/backbones and report what works for Arabic–English OCR + reasoning.
</blockquote>

<hr/>

<h2>🧱 Training Pipeline (stage-wise)</h2>
<p align="center">
  <img src="docs/assets/Train_Pipeline.png" width="900" alt="Roa’ya-VL training pipeline">
</p>
<p align="center"><i>
Base build → Stage-2 SFT (broad instruction mix) → Stage-2.5 Arabic consolidation → Stage-3 preference optimization (when ready).
</i></p>

<hr/>

<h2>📈 Stage-2 Validation (reported)</h2>
<p align="center">
  <img src="docs/assets/stage2_trends_en.png" width="1100" alt="Stage-2 validation trajectory">
</p>
<p align="center"><i>
Validation trend across Stage-2 checkpoints (e.g., MMBench-EN, OCRBench, TextVQA, Avg). Best checkpoint (~45K) is used as Stage-2 reference.
</i></p>

<hr/>

<h2>🏗️ Stages (How we structure development)</h2>

<h3>Stage-2 — Supervised Instruction Tuning (SFT)</h3>
<ul>
  <li><b>Goal:</b> a strong general bilingual VLM base (OCR + reasoning + instruction-following)</li>
  <li><b>Approach:</b> broad multi-domain instruction mix (Arabic included) + checkpoint selection via validation trajectory</li>
</ul>

<h3>Stage-2.5 — Arabic Consolidation (targeted SFT)</h3>
<ul>
  <li><b>Goal:</b> improve Arabic robustness without collapsing general capability</li>
  <li><b>Approach:</b> Arabic OCR/Doc + Arabic instruction + optional culture-focused content, with careful sampling to reduce overfitting/forgetting</li>
  <li><b>Practical note:</b> Arabic gains can be delayed and uneven (OCR vs reasoning vs instruction following)</li>
</ul>

<h3>Stage-3 — Preference Optimization (when data exists)</h3>
<ul>
  <li><b>Goal:</b> improve helpfulness, formatting, reduce hallucinations, and better align Arabic outputs</li>
  <li><b>Approach:</b> DPO / ORPO / GRPO depending on preference/reward data availability</li>
</ul>

<hr/>

<h2>🧪 Evaluation</h2>
<p>
We evaluate checkpoints using standardized tooling (e.g., <b>VLMEvalKit</b>) to track progress across stages.
We recommend evaluating frequently during Stage-2.5 (e.g., every 500–1000 steps) because improvements may appear slowly and differ by skill.
</p>

<hr/>

<h2>📦 Resources</h2>
<ul>
  <li><b>Project page:</b> https://yakoubbazi.github.io/Roaya-VL/</li>
  <li><b>Code:</b> https://github.com/yakoubbazi/Roaya-VL</li>
  <li><b>Models:</b> internal for now (release plan will be announced)</li>
</ul>

<hr/>

<h2>🗺️ Roadmap</h2>
<ul>
  <li>[x] Stage-2 SFT to ~50K steps</li>
  <li>[x] Select best checkpoint (~45K)</li>
  <li>[ ] Expand evaluation (MMStar, MMMU, POPE, RealWorldQA, etc.)</li>
  <li>[ ] Stage-2.5 Arabic consolidation (ongoing)</li>
  <li>[ ] Stage-3 preference optimization (DPO/ORPO/GRPO) when preference data is ready</li>
  <li>[ ] Teaser demos (OCR / Doc / VQA / multi-image)</li>
  <li>[ ] Reproducibility checklist (scripts + configs)</li>
  <li>[ ] Public release (weights + evaluation)</li>
</ul>

<hr/>

<h2>🙏 Acknowledgement</h2>
<ul>
  <li><b>LLaVA-NeXT</b> (training/evaluation utilities and engineering baseline)<br/>
    https://github.com/LLaVA-VL/LLaVA-NeXT
  </li>
  <li><b>DeepSeek-OCR</b> (used in the current reference build; OCR-centric inspiration)<br/>
    https://github.com/deepseek-ai/DeepSeek-OCR
  </li>
  <li><b>VLMEvalKit</b> (evaluation toolkit integration)<br/>
    https://github.com/open-compass/VLMEvalKit
  </li>
</ul>

<blockquote>
  Note: Roa’ya-VL is an independent project and is not affiliated with the above repositories.
</blockquote>

<hr/>

<h2>📌 Citation (project)</h2>
<pre><code class="language-bibtex">
@misc{roaya_vl_framework_2026,
  title   = {Roa'ya-VL: A Practical Framework for Arabic--English Vision-Language Models},
  author  = {Bazi, Yakoub and Zuair, Mansour and Al Rahhal, Mohamad Mahmoud},
  year    = {2026},
  note    = {Project repository and technical notes (proof-of-concept)}
}
</code></pre>
