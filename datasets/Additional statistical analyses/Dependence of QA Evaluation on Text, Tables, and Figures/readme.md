# Dependence of QA Evaluation on Text, Tables, and Figures

To further verify the reliance of QA evaluation on tables and figures, we randomly sampled 500 questions from the test set of **EviMMQA** and manually annotated all QA pairs according to their dependence on text, tables, figures, or combinations thereof. 

| Type         | Count | Percentage (%) |
| ------------ | ----- | -------------- |
| text         | 404   | 80.8           |
| table        | 59    | 11.8           |
| chart        | 20    | 4.0            |
| text + table | 11    | 2.2            |
| text + chart | 6     | 1.2            |

---

We found that approximately **80%** of the QA pairs require only textual understanding, as most key information is described in the text by the review authors, while the remaining **20%** depend on information presented in tables or figures. These findings highlight the necessity of incorporating full-text multimodal content to ensure rigorous QA evaluation.


We then analyzed the articles corresponding to the **70 questions** that involved tables. These 70 questions came from 70 articles and involved a total of **205 tables**.

For each table, we recorded:
(i) the **table type**, and
(ii) the **type of table referenced** by each QA question.


| Table Type                              | Total (Simple : Complex) | QA Pairs Referencing (Simple : Complex) |
| --------------------------------------- | ------------------------ | --------------------------------------- |
| (a) Inclusion/Exclusion Criteria        | 5 (4 : 1)                | 1 (0 : 1)                               |
| (b) Baseline Description                | 70 (48 : 22)             | 47 (25 : 22)                            |
| (c) Outcome Description (no subgroup)   | 63 (48 : 15)             | 19 (10 : 9)                             |
| (d) Outcome Description (with subgroup) | 67 (34 : 33)             | 3 (1 : 2)                               |
| **Total**                               | **205 (134 : 71)**       | **70 (36 : 34)**                        |

---
