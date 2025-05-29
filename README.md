# EviMMQA: Multimodal Question Answering for Medical Evidence Extraction in Systematic Reviews

EviMMQA is a large-scale multimodal dataset designed for complex question answering (QA) tasks in medical evidence extraction, particularly for systematic reviews. Unlike prior datasets focused on abstracts, EviMMQA incorporates full-text articles and supports reasoning across text, tables, and charts.

This repository includes:

- A curated dataset of multimodal medical articles and multiple-choice QA pairs
- A novel framework for evidence extraction QA
- Training and evaluation scripts for multimodal QA models based on LLaVA-NeXT


## 📂 Dataset

- `question.json`: A collection of multiple-choice QA pairs generated from Cochrane systematic reviews.
- `articles/`: Directory containing full-text medical documents

## ⚙️ Usage

### 1. Install environment

We recommend using Anaconda:

```bash
conda create -n evimmqa python=3.10
conda activate evimmqa
pip install -r requirements.txt
```

## 🚀 2. Train the model

```bash
bash scripts/train/train.sh
```

---

## 📊 3. Evaluate the model

```bash
bash scripts/eval/eval.sh
```

---

## 📌 4. To Do List

- [ ] 📦 Release dataset
- [ ] 🧪 Release evaluation scripts
- [ ] 💾 Upload pretrained checkpoints (0.5B / 7B variants)
- [ ] 🏗️ Publish training scripts
- [ ] 📖 Extend to open-ended QA (future)

---

## 🙏 Acknowledgements

This project builds on and extends:

- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

Thanks to **Cochrane Collaboration** and **PubMed Central** for high-quality medical review datasets.

