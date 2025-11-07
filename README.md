## Incentive-Compatible Societies Workspace
End-to-end research sandbox exploring whether audit-and-sanction prompts can elicit truthful meta-knowledge from LLM agents on adversarial factual questions (TruthfulQA). All experimentation was executed in a fresh `uv` virtual environment with local open-weight inference.

### Key Findings
- Sanction prompts *reduced* mean confidence (0.88 → 0.81) and increased abstention (+5 pp) but also dropped accuracy (25% → 15.6%) and worsened calibration (Brier +0.034, ECE +0.058).
- Penalty incidence did not change (67.5% hit-rate in both arms) despite explicit warnings about audits, suggesting prompt-only sanctions lack credibility.
- High-confidence myths (flag burning illegality, ambulance triage, etc.) persisted across protocols, indicating knowledge gaps dominate incentive framing for a 1.5B model.

See `REPORT.md` for full methodology, statistics, and plots.

### Repository Layout
```
data/
  truthfulqa_mc1_sample.json   # 40-question subset used in all runs
results/
  raw/                         # JSONL generations per protocol
  metrics.json                 # Aggregated metrics & statistical tests
  plots/                       # performance_bar.png, confidence_histogram.png, calibration_curve.png
  aggregated_outputs.csv       # Parsed model outputs for analysis
  error_cases.md               # High-confidence mistakes
src/
  incentive_pipeline.py        # Main experiment harness (runs both protocols)
  analyze_results.py           # Post-processing and visualization script
resources.md, planning.md, REPORT.md, README.md
```

### Reproduction
1. **Setup**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```
2. **Run experiments**
   ```bash
   python src/incentive_pipeline.py
   ```
   - Downloads `Qwen/Qwen2.5-1.5B-Instruct`, runs both prompts, saves raw generations + `results/metrics.json`.
3. **Generate analysis artifacts**
   ```bash
   python src/analyze_results.py
   ```
   - Produces plots, `aggregated_outputs.csv`, and `results/error_cases.md`.

Environment notes:
- CPU-only; expect ~2.5 minutes per protocol on AMD EPYC 7302 CPUs.
- All dependencies pinned in `pyproject.toml` / `uv.lock`.

### References
- Lin et al., 2021. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.*
- Gardelli et al., 2006. *Engineering the Environment of Multi-Agent Systems.*
- ACL 2024. *Calibration-Tuning: Teaching Large Language Models to Know What They Don’t Know.*
