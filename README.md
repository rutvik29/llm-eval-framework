# 📏 LLM Eval Framework

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)](https://python.org)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.1-FF6B35?style=flat)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Comprehensive LLM evaluation framework** — RAGAS RAG metrics, hallucination detection, safety testing, bias analysis, and automated regression benchmarks with CI/CD integration.

## ✨ Highlights

- 📊 **RAGAS metrics** — faithfulness, answer relevancy, context recall, context precision
- 🕵️ **Hallucination detection** — NLI-based fact verification against source documents
- 🛡️ **Safety testing** — jailbreak attempts, toxic content, PII leakage detection
- 📉 **Regression benchmarks** — catch quality drops before they reach production
- 🔗 **CI/CD integration** — GitHub Actions workflow with pass/fail gates
- 📈 **Dashboard** — Streamlit dashboard with historical trend analysis

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Faithfulness | Is answer grounded in context? | >0.85 |
| Answer Relevancy | Does answer address the question? | >0.80 |
| Context Recall | Does context contain the answer? | >0.75 |
| Hallucination Rate | % of ungrounded claims | <0.10 |
| Safety Score | Resistance to adversarial inputs | >0.95 |

## Quick Start

```bash
git clone https://github.com/rutvik29/llm-eval-framework
cd llm-eval-framework
pip install -r requirements.txt

# Evaluate a RAG pipeline
python evaluate.py --config configs/rag_eval.yaml --output results/

# Run safety suite
python safety_eval.py --model gpt-4o --suite full

# Launch dashboard
streamlit run dashboard/app.py
```

## License
MIT © Rutvik Trivedi
