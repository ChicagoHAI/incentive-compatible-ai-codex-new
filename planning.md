## Research Question
Can incentive-aware communication protocols—combining mandatory uncertainty reports, sanction penalties, and audit hooks inspired by Predictive Safety Networks—improve the calibration and abstention behaviour of multi-agent LLMs answering high-stakes factual queries (TruthfulQA) relative to a baseline protocol without explicit incentive engineering?

## Background and Motivation
- **Gap**: Standard prompting often yields overconfident hallucinations (Lin et al., 2021). Real deployments require explicit mechanism design so that truthful uncertainty reports are equilibrium strategies.
- **Prior art**: Gardelli et al. (2006) show that engineering the MAS environment (communication/sanction rules) shapes global behaviour. Recent calibration work (e.g., Calibration-Tuning, ACL 2024) demonstrates that carefully designed training/inference protocols can reduce overconfidence. Predictive Safety Network ideas advocate periodic audits and abstention zones for safety-critical settings.
- **Motivation**: By operationalizing these ideas in a concrete evaluation harness, we can produce empirical evidence that even prompt-level environment contracts materially change LLM self-reporting, informing future mechanism design.

## Hypothesis Decomposition
1. **H1 (Calibration)**: Adding audit-backed sanction messaging lowers the Brier score / ECE of self-reported confidence relative to the baseline prompt on the same questions.
2. **H2 (Safety/Abstention)**: Sanctioned prompts yield a higher, purposeful abstention rate on low-certainty items without reducing overall correctness significantly.
3. **H3 (Penalty alignment)**: Expected penalty per question (confidence ≥ τ yet answer wrong) drops under the sanctioned protocol, demonstrating incentive alignment.

Independent variables: prompt protocol (baseline vs. sanction).  
Dependent variables: accuracy, Brier score, ECE, abstention frequency, penalty magnitude.  
Success criteria: observe statistically significant (p < 0.05 Wilcoxon) reductions in Brier score and penalty rate, plus non-trivial abstention adoption.

## Proposed Methodology

### Approach
- Use the TruthfulQA multiple-choice validation set to supply ground-truth audits.
- Run a contemporary open-weight instruct model (`Qwen/Qwen2.5B-Instruct`) locally to ensure genuine LLM behaviour without API keys.
- Design two prompts that differ only by environment-contract framing; both demand JSON outputs (`choice`, `confidence`, optional `abstain`).
- Implement a sanction scoring function emulating Predictive Safety Networks: random audits with transparency around penalties for overstated confidence.
- Compare metrics using paired analyses because both prompts receive identical question subsets.

### Experimental Steps
1. **Data curation**: Load TruthfulQA (multiple_choice) via `datasets`, stratified sample of 40 Qs ensuring topical diversity; store split metadata.
2. **Prompt harness**: Build reusable generator that formats question/options and enforces JSON-only responses with retry-on-parse failure.
3. **Baseline runs**: Query the model with the neutral prompt, log outputs (`results/raw/baseline.jsonl`).
4. **Sanction runs**: Query with incentive-aware prompt, log outputs (`results/raw/sanction.jsonl`).
5. **Metrics computation**: Parse JSON, convert answers to correctness labels, compute accuracy, abstention rate, Brier score, 10-bin ECE, penalty stats, and Wilcoxon tests.
6. **Visualization & analysis**: Produce calibration plots/histograms, penalty boxplots, and summarise findings in REPORT.md.

### Baselines
- **Neutral prompt baseline**: Standard QA instructions with explicit request for best answer + probability. Acts as control for natural calibration.
- **Random guess sanity check** (optional): Compute expected accuracy/Brier for uniform random answers to contextualize improvements.

### Evaluation Metrics
- **Accuracy**: share of correct answers excluding abstentions; ensures truthfulness not sacrificed.
- **Brier score**: mean squared error between confidence and correctness indicator.
- **Expected Calibration Error (ECE)**: 10-bin absolute gap between mean confidence and accuracy.
- **Penalty rate/size**: Share of audited items incurring penalties and average penalty magnitude (confidence when wrong).
- **Abstention rate**: Fraction of samples where the model elects to abstain.
- Statistical significance via paired Wilcoxon on Brier/penalty differences; report bootstrap CIs for accuracy delta.

### Statistical Analysis Plan
- Use identical question ordering to obtain paired samples.
- Apply Wilcoxon signed-rank for non-parametric confidence/Brier comparisons (n≈40).
- Bootstrap (10k resamples) the accuracy difference and abstention change to derive 95% CIs.
- Control for multiple hypotheses with Holm-Bonferroni across H1–H3.

## Expected Outcomes
- Sanctioned protocol should reduce overconfidence (lower Brier/ECE) and introduce modest abstention plus lower penalty rates.
- If no improvement appears, document failure cases to refine mechanism design; negative result still informative.

## Timeline and Milestones (≈3h budget)
1. **0:00–0:35** – Finish planning, install deps, load dataset, craft prompts.
2. **0:35–1:30** – Implement harness + run baseline & sanction generations (80 calls total, monitor runtime).
3. **1:30–2:15** – Compute metrics, statistical tests, generate plots.
4. **2:15–3:00** – Draft REPORT.md & README.md, finalize validation checklist.
Include ~20% buffer for parsing/debugging.

## Potential Challenges & Mitigations
- **Slow inference on CPU**: Use small sample size (≤40) and low `max_new_tokens`; leverage caching of tokenizer/model objects.
- **JSON parsing errors**: Add regex extraction and limited retries with corrective prompts.
- **Lack of API keys**: Already mitigated by using open-weight local models.
- **Calibration noise**: Use paired tests + bootstrap to provide uncertainty estimates despite small sample.

## Success Criteria
- Complete logs + metrics demonstrating measurable change between protocols.
- REPORT.md summarizing statistically analysed results and limitations.
- Reproducible pipeline (seeded sampling, saved configs, instructions in README).
