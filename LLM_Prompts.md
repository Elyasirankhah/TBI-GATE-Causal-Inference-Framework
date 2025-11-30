# LLM Counterfactual Validation Prompts

This document contains the exact prompts used for LLM-based counterfactual prediction in our TBI rehabilitation study.

---

## 🎯 Overview

The LLM counterfactual validation uses Meta-Llama-3.1-8B-Instruct to predict patient outcomes under different treatment scenarios. This approach provides an independent validation of our causal inference results.

---

## 📝 Data Augmentation Prompts

### Purpose
Synthetic patient sequence generation for underrepresented subgroups

### Model
DistilGPT-2 (tbims_medical_embeddings); same model used for embeddings

### Template: PROMPT_TEMPLATE

```
You are generating synthetic PRE-INDEX clinical sequences for traumatic brain injury patients.
Return ONLY a single line of tokens in the SAME format and style as the example, with NO explanations.

Constraints:
- Use SDOH header tokens first: {AGE_TOK} {SEX_TOK} {RACE_TOK} {ETH_TOK}
- Then a SPACE-separated list of pre-index event tokens in the EXACT compact format:
  {DOM}:{CODE}|{GAP}|V{TAG}
- DO NOT include any post-index events or outcomes, and DO NOT include treatment indicators ({TREAT}).
- Keep total tokens under {MAX_TOKENS} tokens.

Patient anchor (style/example; do NOT copy verbatim, generate a plausible variant):
{ANCHOR_TEXT}

Augmentation goals for this subgroup:
- Maintain subgroup identity: {SUBGROUP_DESC}
- Vary plausible pre-index conditions/procedures/drugs and temporal gaps to enrich coverage.
- Keep medical plausibility and diversity; avoid duplicating the anchor sequence.

Return ONLY the token sequence, nothing else.
```

### Results
Generated 9,600 structured prompts

---

## 🔄 Counterfactual Validation Prompts

### Purpose
Counterfactual outcome prediction for overlap cohort validation

### Model
Meta-Llama-3.1-8B-Instruct

### System Prompt

```
You are a careful clinical assistant. Return ONLY valid JSON with numeric fields p1 and p0 in [0,1]. 
Important: p1 and p0 are INDEPENDENT probabilities under different interventions. 
DO NOT set p0 = 1 - p1 or vice versa.
```

### User Prompt Template: USER_TMPL

```
TBI patient profile (pre-index only): {profile}
Outcome: {outcome}
Clinical prior (population-level): observed {outcome} rate in similar patients is ~{base_rate:.1f}%.

Estimate two independent probabilities for this specific patient:
- p1 = P({outcome} | do(T=1: early rehabilitation within {K} days))
- p0 = P({outcome} | do(T=0: no early rehab within {K} days))

Constraints:
- p1 and p0 must be in [0,1].
- p1 and p0 are NOT complements; do not force p1 + p0 = 1.
- Anchor, but do not copy, the population rate {base_rate:.1f}% when appropriate.
- Use two decimals if needed.

Return strict JSON only: {"p1": <0..1>, "p0": <0..1>}
```

### Validation
Consistency with IPTW/AIPW estimates

### Calibration
5-fold isotonic regression on factual arms

---

## ⚙️ Prompt Configuration

- **Temperature:** 0.35
- **Top-p:** 0.95
- **Max tokens:** 64
- **Batch size:** 16
- **Total calls:** 1,046 (1 outcome, 1 sample per patient)

---

---

## 🎯 Usage Instructions

1. **Prepare Patient Data**: Ensure all required demographic and clinical variables are available
2. **Generate Summaries**: Create patient-specific summaries using the templates
3. **Run LLM Inference**: Submit prompts to Meta-Llama-3.1-8B-Instruct
4. **Parse Responses**: Extract probabilities and reasoning from responses
5. **Apply Calibration**: Use isotonic regression to calibrate predictions
6. **Validate Results**: Compare with traditional causal inference methods

---

## 📈 Results Summary

### Phase 3 LLM Validation Results

- **Cohort Size**: 1,046 patients (523 matched pairs)
- **Raw LLM ATE**: +0.0394 (shows harm - expected before calibration)
- **Calibrated LLM ATE**: -0.1089 (shows benefit - final result)
- **IPTW ATE**: -0.0767 (traditional method)
- **Alignment**: Both methods show benefit (negative ATE)


---

## 📚 References

- Meta-Llama-3.1-8B-Instruct: [Hugging Face Model](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
- Isotonic Regression: [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)
- Prompt Engineering: [Best Practices Guide](https://www.promptingguide.ai/)

---

## 📧 Contact

For questions about LLM prompt design or counterfactual validation:

**Corresponding Author:** [Elyas Irankhah]
- Email: Elyas_irankhah@student.uml.edu
- ORCID: [0000-0001-7168-3898]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

