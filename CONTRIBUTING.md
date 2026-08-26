# Contributing

Contributions are welcome. This project benefits from improvements to detection accuracy, new feature families, expanded test coverage, and documentation.

## Getting Started

```bash
git clone https://github.com/A-Kuo/Natural-Hallucination-Analysis.git
cd Natural-Hallucination-Analysis
pip install -e ".[dev]"
```

## Running Tests

```bash
# pytest suite
pytest tests -v

# module self-tests
python feature_engineer.py
python detector.py
python data_generator.py
python entropy_baselines.py
python calibrated_entropy_detector.py
python blackbox_detector.py
python adversarial.py

# synthetic pipeline
python pipeline.py --synthetic --num_samples 500
```

## Pull Request Guidelines

1. **Fork** the repo and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full test suite** before submitting.
4. **Keep commits focused** — one logical change per commit.
5. **Update documentation** if your change affects the public API or agent instructions (`AGENT.md`).

## Code Style

- Python 3.10+
- Type hints on all public functions
- Docstrings on all public classes and functions
- NumPy-style docstrings preferred
- Core detection code stays open-source-model-friendly (EleutherAI, Meta, Mistral); the only OpenAI dependency is the optional, lazy-imported `blackbox_detector.py::fetch_topk_logprobs_openai()` demo path — never make OpenAI a hard dependency of any test or the default pipeline

## Adding a New Feature Family

1. Implement `compute_<family>_features()` in `feature_engineer.py`
2. Add the family to `FeatureConfig` and `FEATURE_SIZES`
3. Add a self-test block in the `__main__` section, plus real pytest coverage in `tests/`
4. Update `AGENT.md` with the mathematical foundation
5. Run the synthetic pipeline to verify integration

## Reporting Issues

Use [GitHub Issues](https://github.com/A-Kuo/Natural-Hallucination-Analysis/issues). Include:
- Python version and OS
- Minimal reproduction steps
- Full error traceback
