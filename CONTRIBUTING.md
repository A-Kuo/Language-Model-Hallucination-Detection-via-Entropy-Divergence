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
# v1 pytest suite
pytest v1/test_attention_analyzer.py -v

# v1 module self-tests
python v1/hypothesis_test.py
python v1/confidence_calibrator.py

# v2 self-tests
python v2/feature_engineer.py
python v2/detector.py
python v2/data_generator.py

# v2 synthetic pipeline
python v2/pipeline.py --synthetic --num_samples 500
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
- No OpenAI model dependencies — use open-source models (EleutherAI, Meta, Mistral)

## Adding a New Feature Family (v2)

1. Implement `compute_<family>_features()` in `v2/feature_engineer.py`
2. Add the family to `FeatureConfig` and `FEATURE_SIZES`
3. Add a self-test block in the `__main__` section
4. Update `v2/AGENT.md` with the mathematical foundation
5. Run the synthetic pipeline to verify integration

## Reporting Issues

Use [GitHub Issues](https://github.com/A-Kuo/Natural-Hallucination-Analysis/issues). Include:
- Python version and OS
- Minimal reproduction steps
- Full error traceback
