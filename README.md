# EviMMQA: Multimodal Question Answering for Medical Evidence Extraction in Systematic Reviews

EviMMQA is a large-scale multimodal dataset designed for complex question answering (QA) tasks in medical evidence extraction, particularly for systematic reviews. Unlike prior datasets focused on abstracts, EviMMQA incorporates full-text articles and supports reasoning across text, tables, and charts.

This repository includes:

- A curated dataset of multimodal medical articles and multiple-choice QA pairs
- A novel framework for evidence extraction QA
- Training and evaluation scripts for multimodal QA models based on LLaVA-NeXT

---

## ⚙️ Environment Setup

We recommend using **Anaconda**:

```bash
conda create -n evimmqa python=3.10
conda activate evimmqa
pip install -r requirements.txt
```

Download the pretrained model:

```
lmms-lab/llava-next-interleave-qwen-7b-dpo
```

---
## 📂 Dataset

1. Download the dataset from [Hugging Face Hub](https://huggingface.co/changkai/EviMMQA):
   ```
   changkai/EviMMQA
   ```

2. Unzip the files:
   ```bash
   unzip articles1.zip
   unzip articles2.zip
   ```

3. Enter the dataset processing directory:
   ```bash
   cd dataset_processing
   ```

4. For both `train.json` and `test.json`, run the following steps:
   ```bash
   python paddle_ocr.py
   python prepare_rag.py
   python gpt_format_transform.py
   ```

---

## 🚀 Training

Two training scripts are supported:

- **Standard fine-tuning**
  ```bash
  bash ft.sh
  ```

- **MoE fine-tuning**
  ```bash
  bash ft_moe.sh
  ```

---

## 📊 Evaluation

Run evaluation with:

```bash
python eval_pico.py
```

---

## 🙏 Acknowledgements

This project builds on and extends:

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

Thanks to **Cochrane Collaboration** and **PubMed Central** for high-quality medical review datasets.

