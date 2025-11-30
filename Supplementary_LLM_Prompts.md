# Supplementary File S1 — Unified LLM Prompts and Configuration

This supplementary file contains all system prompts, user prompts, templates, and configuration settings used in Contribution 3 (LLM Validation) of the TBI-GATE Causal Inference Framework. The full prompt set (data augmentation + counterfactual validation) is combined below in one unified block exactly as used in the framework. All patient profiles consist solely of de-identified, pre-index features, with no PHI.

--------------------------------------------------------------------------------
# FULL COMBINED PROMPT BLOCK (ALL PROMPTS IN ONE PLACE)

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

--------------------------------------------------------------------------------

You are a careful clinical assistant. Return ONLY valid JSON with numeric fields p1 and p0 in [0,1]. 
Important: p1 and p0 are INDEPENDENT probabilities under different interventions. 
DO NOT set p0 = 1 - p1 or vice versa.

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

--------------------------------------------------------------------------------
# CONFIGURATION USED FOR ALL LLM CALLS

Model for data augmentation: DistilGPT-2 (tbims_medical_embeddings)  
Model for counterfactual validation: Meta-Llama-3.1-8B-Instruct  

Generation settings (applied universally):  
- Temperature: 0.35  
- Top-p: 0.95  
- Max tokens: 64  
- Batch size: 16  
- Total calls: 1,046 (one patient × one outcome × one counterfactual pair)

Calibration:  
- 5-fold isotonic regression applied separately to factual treated and control arms before computing calibrated effects.

Quality filtering for augmented sequences:  
- Total generated: 9,600  
- Passed automated filters: 1,198  
- Passed manual plausibility review: 892  

--------------------------------------------------------------------------------
# END OF SUPPLEMENTARY FILE S1
