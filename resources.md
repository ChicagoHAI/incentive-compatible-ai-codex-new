## Resource Scan Summary

- **Workspace contents**: Only scaffolding folders (`artifacts/`, `logs/`, `notebooks/`, `results/`) plus the freshly created `pyproject.toml`. No datasets, notebooks, or reports were supplied, so all assets must be produced in this session.
- **Key gaps**: No dataset, baseline implementation, or evaluation harness provided; no documentation beyond the system instructions. Need to source benchmark questions, select an LLM, and design incentive-compatible prompting plus analysis code.

## Literature & Method References

| Topic | Resource | Key Takeaways |
|-------|----------|---------------|
| Truthfulness benchmarks | [TruthfulQA paper (Lin et al., 2021)](https://arxiv.org/abs/2109.07958) via DuckDuckGo | 817 expert-written questions spanning 38 categories highlight hallucination-prone areas; standard choice for evaluating truthful generation. |
| Environment engineering | Gardelli et al. 2006, “Engineering the Environment of Multi-Agent Systems” ([PDF](http://unibo.lgardelli.com/publications/gardelli2006d.pdf)) | Formalizes shaping the MAS environment (communication rules, sanctions) to induce desired equilibria—supports our mechanism-design framing. |
| Calibration techniques | “Calibration-Tuning: Teaching Large Language Models to Know What They Don’t Know” (ACL 2024, [ACL Anthology](https://aclanthology.org/2024.uncertainlp-1.1/)) | Uses protocol tweaks to reduce overconfidence without hurting accuracy; motivates measuring Brier/Expected Calibration Error for sanction prompts. |
| Predictive safety | Public discussion of Predictive Safety Networks (e.g., [PRISM overview](https://prism.sustainability-directory.com/term/predictive-safety-networks/)) | Describes proactive monitoring hooks; inspires the “audit hook” prompts that sample spot checks and penalize misreports. |

These sources justify combining environment design (communication rules, sanction contracts) with truthfulness/cross-check auditing metrics from LLM calibration literature.

## Dataset Decision

- **Chosen benchmark**: `truthfulqa/truthful_qa` (multiple_choice config) from Hugging Face. Small (<1 MB), CPU-friendly, includes ground-truth labels for quick auditing. Licensed Apache-2.0, so redistributable.
- **Usage plan**: Sample ~40 questions for rapid iteration while preserving topical diversity via stratified sampling across categories. Offers binary correctness for calibration metrics and supports computing penalties when confidence claims mismatch ground truth.

## Model & Tooling Decision

- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (released 2024, instruction-tuned). Runs locally on CPU via `transformers` + `torch`, giving us a *real* pretrained model despite lacking hosted API keys in this environment.
- **Rationale**: Instructions require real LLM behaviour. No `OPENAI_API_KEY`/`OPENROUTER_API_KEY` variables are present (confirmed via `env | grep`), so local inference with contemporary open weights is the most viable path while honoring the “no simulations” constraint.
- **Supporting libraries**: `datasets` for loading TruthfulQA, `pandas`/`numpy` for tabular handling, `matplotlib`/`seaborn` for plots, `scipy` for statistical tests, `tqdm` for progress, `python-dotenv` optional if secrets appear later.

## Baseline & Mechanism Concepts

1. **Baseline protocol**: Standard QA instruction asking for best answer + calibrated confidence (JSON output). Serves as control for natural overconfidence.
2. **Sanctioned protocol**: Extended prompt encoding environment rules—mandatory uncertainty reports, abstention option, periodic audits, and penalties for overstated confidence. Will log penalties when confidence ≥ τ yet answer is wrong.

## Metrics & Evaluation Targets

- Accuracy, macro-averaged.
- Calibration quality: Brier score and Expected Calibration Error (ECE, 10-bin).
- Incentive alignment: frequency/size of penalty events, abstention rate, and audit loss (mean penalty).
- Statistical comparison: Paired bootstrap or Wilcoxon signed-rank between protocols across identical question set.

## Next Research Needs

- Implement reproducible prompting harness (JSON parsing, retries).
- Define audit sampling/penalty function consistent with Predictive Safety Network ideas (e.g., random audits at rate p, penalty scales with confidence gap).
- Store raw generations per protocol for traceability (`results/raw/…jsonl`).

This resources scan completes Phase 0; proceed to detailed planning (Phase 1) using the assets above.
