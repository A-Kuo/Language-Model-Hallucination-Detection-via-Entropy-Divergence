"""
Utilities — Helpers for the Hallucination Detection Pipeline
=============================================================

Provides:
    - TokenizationHelper: wraps HuggingFace tokenizer with convenience methods
    - setup_logger: structured logging with consistent formatting
    - Timer: context manager for profiling code blocks
    - serialize_result / deserialize_result: JSON I/O for analysis results
    - batch_texts: chunk a list of strings for memory-efficient processing

Usage:
    from src.utils import setup_logger, Timer, TokenizationHelper

    logger = setup_logger("hallucination_detection")
    with Timer("forward_pass", logger=logger):
        result = analyzer.analyze(text)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(
    name: str = "hallucination_detection",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    fmt: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
) -> logging.Logger:
    """
    Create a structured logger with console (and optional file) output.

    Parameters
    ----------
    name : str
        Logger name (appears in log lines).
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_file : str | Path | None
        If set, also write logs to this file.
    fmt : str
        Log line format string.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        fh = logging.FileHandler(str(log_file), mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager and decorator for timing code blocks.

    Usage as context manager:
        with Timer("forward_pass") as t:
            result = model(**inputs)
        print(t.elapsed_ms)

    Usage as decorator:
        @Timer.decorator("my_function")
        def expensive_fn():
            ...
    """

    def __init__(
        self,
        label: str = "block",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.label = label
        self.logger = logger
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        if self.logger:
            self.logger.info(f"[Timer] {self.label}: {self.elapsed_ms:.2f} ms")

    @staticmethod
    def decorator(
        label: str = "function",
        logger: Optional[logging.Logger] = None,
    ):
        """Use Timer as a function decorator."""
        def wrapper(fn):
            def inner(*args, **kwargs):
                with Timer(label, logger=logger):
                    return fn(*args, **kwargs)
            inner.__name__ = fn.__name__
            inner.__doc__ = fn.__doc__
            return inner
        return wrapper


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

class TokenizationHelper:
    """
    Wraps a HuggingFace tokenizer with convenience methods.
    Lazily loads the tokenizer on first use.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.tokenizer.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int = 1024) -> str:
        """Truncate *text* to at most *max_tokens* tokens."""
        ids = self.tokenizer.encode(text, truncation=True, max_length=max_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def split_into_chunks(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> List[str]:
        """
        Split *text* into overlapping token-level chunks.
        Useful for processing long documents exceeding model context.
        """
        ids = self.tokenizer.encode(text)
        chunks: List[str] = []
        step = max(1, chunk_size - overlap)

        for start in range(0, len(ids), step):
            chunk_ids = ids[start:start + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)

            if start + chunk_size >= len(ids):
                break

        return chunks


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def batch_texts(
    texts: List[str],
    batch_size: int = 32,
) -> Iterator[List[str]]:
    """
    Yield successive batches of *batch_size* from *texts*.

    Usage:
        for batch in batch_texts(my_texts, batch_size=16):
            results = analyzer.analyze_batch(batch)
    """
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


# ---------------------------------------------------------------------------
# JSON serialization for results
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_result(result: Any, path: Union[str, Path]) -> None:
    """
    Save a dataclass result to JSON.
    Handles numpy arrays/types. Strips raw_attentions (too large).
    """
    d = _dataclass_to_dict(result)
    d.pop("raw_attentions", None)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, cls=NumpyEncoder, indent=2)


def deserialize_result(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON result file into a dictionary."""
    with open(path, "r") as f:
        return json.load(f)


def _dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass to dict, handling nested dataclasses and enums."""
    if hasattr(obj, "__dataclass_fields__"):
        d = {}
        for field_name in obj.__dataclass_fields__:
            val = getattr(obj, field_name)
            d[field_name] = _dataclass_to_dict(val)
        return d
    elif hasattr(obj, "value"):  # Enum
        return obj.value
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Metrics formatting
# ---------------------------------------------------------------------------

def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Format a dict of metrics as an ASCII table."""
    if not metrics:
        return f"  {title}: (empty)"

    max_key = max(len(str(k)) for k in metrics)
    max_val = max(len(f"{v:.4f}" if isinstance(v, float) else str(v))
                  for v in metrics.values())

    col_k = max(max_key, len("Metric"))
    col_v = max(max_val, len("Value"))

    lines = [
        f"┌{'─' * (col_k + col_v + 5)}┐",
        f"│ {title:<{col_k + col_v + 3}} │",
        f"├{'─' * (col_k + 2)}┬{'─' * (col_v + 2)}┤",
    ]
    for k, v in metrics.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        lines.append(f"│ {str(k):<{col_k}} │ {v_str:>{col_v}} │")
    lines.append(f"└{'─' * (col_k + 2)}┴{'─' * (col_v + 2)}┘")

    return "\n".join(lines)
