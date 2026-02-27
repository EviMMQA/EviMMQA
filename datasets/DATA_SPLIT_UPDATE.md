## Revised Data Partition

To improve the evaluation rigor and practical reliability of the dataset, we have re-partitioned EviMMQA into a standardized:

> **Train : Validation : Test = 6 : 1 : 3**

* The original **30% test set** is preserved.
* The original training pool is split into:

  * 60% Training
  * 10% Validation

### Statistics

| Split      | Documents | Questions | Percentage |
| ---------- | --------- | --------- | ---------- |
| Training   | 9,632     | 168,890   | 60%        |
| Validation | 1,606     | 28,984    | 10%        |
| Test       | 4,817     | 92,178    | 30%        |
| **Total**  | 16,055    | 290,051   | 100%       |

---

## Updated Experimental Results

We re-conducted fine-tuning experiments under the revised protocol for:

* LLaVA-NeXT-Interleave
* Our proposed framework

Model selection was performed **only on validation performance**, and the test set remained strictly held out.

| Model                        | Train | Validation | Test |
| ---------------------------- | ----- | ---------- | ---- |
| LLaVA-NeXT-Interleave (0.5B) | 0.67  | 0.64       | 0.64 |
| Ours (0.5B)                  | 0.76  | 0.72       | 0.71 |
| LLaVA-NeXT-Interleave (7B)   | 0.82  | 0.79       | 0.78 |
| Ours (7B)                    | 0.89  | 0.85       | 0.85 |

Validation and test performance remain highly consistent, indicating stable generalization and no observable overfitting under the standardized protocol.
