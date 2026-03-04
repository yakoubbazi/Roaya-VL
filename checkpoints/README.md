# 📦 Checkpoints

Model checkpoints are **not stored in this repository** in order to keep the repo lightweight.

Please download the official Roaya-VL model weights from:

**BigData-KSU/Roaya-VL-3B**

---

## ⬇️ Download via Command Line

Install the Hugging Face client:

```bash
pip install huggingface_hub
```

Download the model:

```bash
huggingface-cli download BigData-KSU/Roaya-VL-3B --local-dir checkpoints/Roaya-VL-3B
```

The model will be saved inside the `checkpoints/` directory.

---

## ⚠️ DeepSeek-OCR (Vision Tower)

Roaya-VL uses **DeepSeek-OCR Vision encoder based on SAM and CLIP** as the vision encoder.

If you encounter issues loading:

`deepseek-ai/DeepSeek-OCR`

(for example due to **FlashAttention** dependencies), download the model separately and place it locally (e.g., inside `checkpoints/`).

Then update the Roaya-VL configuration file:

```
config.json
```

Example:

```json
"mm_vision_tower": "deepseek-ai/DeepSeek-OCR"
```

If DeepSeek-OCR is stored locally, replace it with the **local path** where you saved the model.
