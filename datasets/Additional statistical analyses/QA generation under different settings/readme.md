## QA Generation from Structured Study Characteristics

This folder evaluates the impact of different hyperparameters on the generation of QA pairs from structured study characteristics using GPT-4o. We conducted a preliminary evaluation to test how different settings of temperature and top-p affect QA generation. Due to space limitations, these results were not included in the main manuscript.

We randomly sampled several study characteristics and generated QA pairs under different hyperparameter configurations of temperature (0, 0.5, 1) and top-p (0.8, 0.95). Two domain experts independently assessed the generated QA pairs along four dimensions:

- Accuracy: Whether the QA pair is factually correct (0 or 1)
- Fluency: Linguistic quality of the generated text (0–10)
- Coverage: Whether each study characteristic yields at least one valid question (0–10)
- Diversity: Variety in the generated QA pairs (0–10)

Final scores were averaged across the two evaluators. The results are reported in the table.

| Temperature | Top-p | Accuracy | Fluency | Coverage | Diversity |
| ----------- | ----- | -------- | ------- | -------- | --------- |
| 0.0         | 0.80  | 1.0000   | 9.8711  | 9.0000   | 7.0000    |
| 0.0         | 0.95  | 0.9947   | 9.8073  | 9.2222   | 6.8889    |
| 0.5         | 0.80  | 0.9946   | 9.8127  | 8.5556   | 6.6667    |
| 0.5         | 0.95  | 0.9947   | 9.8430  | 8.6667   | 6.6667    |
| 1.0         | 0.80  | 1.0000   | 9.8625  | 9.1111   | 7.0000    |
| 1.0         | 0.95  | 1.0000   | 9.8672  | 8.4000   | 6.8889    |


**Observation:**
All metrics remained largely consistent across hyperparameter settings. To ensure reproducibility of QA generation, we therefore chose a temperature of 0 and top-p of 0.95 for our final configuration.  

**Examples:**
Example of QA pair generation-1.png: QA pair generation from Methods characteristics
Example of QA pair generation-2.png: QA pair generation from Participants characteristics

