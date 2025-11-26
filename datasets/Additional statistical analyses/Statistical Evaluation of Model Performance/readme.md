# Statistical Evaluation of Model Performance

This document summarizes the independent repeated experiments, statistical significance testing, and bootstrap-based uncertainty estimation conducted in our study. Due to time constraints, only a representative subset of models is included in the repeated experiments:

* **InternVL-2.5**
* **LLaVA-NeXT-Interleave (7B)**
* **Fine-tuned LLaVA-NeXT-Interleave (7B)**
* **Our method (7B)**

Bootstrap significance estimation, however, was performed on **all models**.

---

## 1. Independent Repeated Experiments

We repeated the experiments **five times** using different random seeds. For each run, we recorded the accuracy (Acc) and report **mean ± standard deviation (SD)** and the **95% confidence interval (CI)**.

### **Table 1. Independent repeated experiments (mean ± SD, 95% CI)**

| Model                                 | All                             | Method                          | Participant                     | Intervention                    | Outcome                         | Context                         | UIQA                            | ESQA                            |
| ------------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- | ------------------------------- |
| InternVL-2.5                          | 84.40 ± 1.95 (81.98, 86.82)     | 87.40 ± 0.55 (86.72, 88.08)     | 83.00 ± 2.92 (79.38, 86.62)     | 85.00 ± 3.39 (80.79, 89.21)     | 83.80 ± 0.84 (82.76, 84.84)     | 79.40 ± 16.49 (58.93, 99.87)    | 77.40 ± 2.97 (73.72, 81.08)     | 84.60 ± 2.07 (82.03, 87.17)     |
| LLaVA-NeXT-Interleave (7B)            | 73.20 ± 1.10 (71.84, 74.56)     | 78.20 ± 1.30 (76.58, 79.82)     | 65.80 ± 2.05 (63.26, 68.34)     | 77.80 ± 2.17 (75.11, 80.49)     | 75.20 ± 1.64 (73.16, 77.24)     | 84.60 ± 9.48 (72.83, 96.37)     | 30.80 ± 2.68 (27.47, 34.13)     | 74.20 ± 1.10 (72.84, 75.56)     |
| Fine-tuned LLaVA-NeXT-Interleave (7B) | 81.00 ± 1.41 (79.24, 82.76)     | 84.80 ± 1.30 (83.18, 86.42)     | 77.20 ± 0.84 (76.16, 78.24)     | 82.60 ± 1.14 (81.18, 84.02)     | 82.20 ± 3.90 (77.36, 87.04)     | 82.00 ± 3.39 (77.79, 86.21)     | 87.40 ± 2.88 (83.82, 90.98)     | 81.00 ± 1.41 (79.24, 82.76)     |
| **Ours (7B)**                         | **84.40 ± 0.89 (83.29, 85.51)** | **88.40 ± 3.21 (84.42, 92.38)** | **80.20 ± 0.84 (79.16, 81.24)** | **86.60 ± 0.89 (85.49, 87.71)** | **84.60 ± 0.89 (83.49, 85.71)** | **88.60 ± 4.98 (82.42, 94.78)** | **93.80 ± 0.45 (93.24, 94.36)** | **84.20 ± 0.84 (83.16, 85.24)** |

**Observation:**
Our model consistently outperforms both the vanilla and fine-tuned LLaVA-NeXT-Interleave models across all subtasks, and achieves comparable performance to InternVL-2.5.

---

## 2. Statistical Significance Testing (McNemar’s Test)

To evaluate whether performance differences are statistically significant, we conducted **McNemar’s test**.

### **2×2 Contingency Table Definition**

```
e_00: both models predict correctly  
e_01: ours correct, baseline incorrect  
e_10: ours incorrect, baseline correct  
e_11: both incorrect  
```

### **McNemar’s Test Statistic**

```text
χ² = (|e_01 - e_10| - 1)² / (e_01 + e_10)
```

If **p < 0.05**, the improvement is statistically significant.

### **Table 2. McNemar’s test results**

| Comparison                                | e_01  | e_10 | χ²      | p-value | Significant? |
| ----------------------------------------- | ----- | ---- | ------- | ------- | ------------ |
| Ours vs. InternVL-2.5                     | 6217  | 5676 | 24.7    | < 0.001 | Yes          |
| Ours vs. LLaVA-NeXT-Interleave            | 12417 | 3768 | 4620.82 | < 0.001 | Yes          |
| Ours vs. Fine-tuned LLaVA-NeXT-Interleave | 7167  | 4527 | 595.55  | < 0.001 | Yes          |

**Conclusion:**
All comparisons yield p-values far below 0.05, confirming that the improvements achieved by our model are statistically significant.

---

## 3. Bootstrap Significance Estimation

We used **1,000 bootstrap samples** (sampling with replacement) to compute:

* Mean accuracy
* Standard error (SE)
* 95% confidence interval (CI)

### **Table 3. Bootstrap accuracy, SE, and 95% CI**

| Model                            | Params | ALL      | Method   | Participant | Intervention | Outcome  | Context  | UIQA     | ESQA     | Acc       | SE        | 95% CI             |
| -------------------------------- | ------ | -------- | -------- | ----------- | ------------ | -------- | -------- | -------- | -------- | --------- | --------- | ------------------ |
| LLaMA-3                          | 8B     | 0.67     | 0.75     | 0.58        | 0.70         | 0.69     | 0.67     | 0.59     | 0.67     | 0.664     | 0.002     | [0.662, 0.667]     |
| Qwen-2.5                         | 7B     | 0.77     | 0.83     | 0.70        | 0.82         | 0.77     | 0.78     | 0.67     | 0.77     | 0.751     | 0.001     | [0.749, 0.754]     |
| InternVL-2.5                     | 8B     | 0.85     | 0.87     | 0.83        | 0.87         | 0.84     | 0.86     | 0.73     | 0.85     | 0.846     | 0.001     | [0.843, 0.848]     |
| InternLM-XComposer2              | 8B     | 0.80     | 0.84     | 0.75        | 0.84         | 0.80     | 0.81     | 0.55     | 0.80     | 0.797     | 0.001     | [0.795, 0.800]     |
| Qwen2-VL                         | 7B     | 0.82     | 0.87     | 0.77        | 0.87         | 0.83     | 0.86     | 0.71     | 0.83     | 0.825     | 0.001     | [0.822, 0.827]     |
| mPLUG-Owl3                       | 7B     | 0.77     | 0.83     | 0.73        | 0.80         | 0.78     | 0.79     | 0.56     | 0.78     | 0.775     | 0.001     | [0.773, 0.778]     |
| LLaVA-v1.6                       | 7B     | 0.78     | 0.84     | 0.70        | 0.83         | 0.79     | 0.81     | 0.47     | 0.78     | 0.778     | 0.001     | [0.775, 0.781]     |
| LLaVA-NeXT-Interleave            | 0.5B   | 0.42     | 0.47     | 0.40        | 0.41         | 0.43     | 0.54     | 0.25     | 0.43     | 0.423     | 0.002     | [0.420, 0.426]     |
| LLaVA-NeXT-Interleave            | 7B     | 0.73     | 0.78     | 0.66        | 0.77         | 0.74     | 0.77     | 0.27     | 0.73     | 0.726     | 0.001     | [0.723, 0.729]     |
| Fine-tuned LLaVA-NeXT-Interleave | 0.5B   | 0.64     | 0.72     | 0.59        | 0.63         | 0.64     | 0.74     | 0.70     | 0.63     | 0.643     | 0.001     | [0.640, 0.648]     |
| Fine-tuned LLaVA-NeXT-Interleave | 7B     | 0.81     | 0.86     | 0.77        | 0.82         | 0.81     | 0.84     | 0.87     | 0.81     | 0.809     | 0.001     | [0.807, 0.812]     |
| **Ours**                         | 0.5B   | 0.72     | 0.80     | 0.69        | 0.72         | 0.72     | 0.84     | 0.88     | 0.72     | 0.725     | 0.001     | [0.722, 0.727]     |
| **Ours**                         | 7B     | **0.85** | **0.90** | **0.80**    | **0.86**     | **0.85** | **0.90** | **0.94** | **0.85** | **0.852** | **0.001** | **[0.850, 0.854]** |

**Findings:**

* Standard errors are extremely small (0.001–0.002).
* Confidence intervals are narrow, indicating statistically reliable results.
* Our model achieves the highest overall accuracy and shows the strongest robustness.

---

## Summary

Across all evaluations—including repeated experiments, statistical significance tests, and bootstrap estimation—our model demonstrates:

* **Consistent and strong performance**
* **Statistically significant improvements over baselines**
* **High robustness and stability across samples**
