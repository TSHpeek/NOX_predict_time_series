"""SEER package exports.

Keep the benchmark adapter lazy so importing the standalone ``SEERModel`` does
not eagerly import every optional baseline dependency (for example,
``reformer_pytorch`` used by Crossformer).
"""

__all__ = ["SEER"]


def __getattr__(name):
    if name == "SEER":
        from ts_benchmark.baselines.seer.seer import SEER

        return SEER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
