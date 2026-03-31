"""
Scanner modules for the Zero-DTE Options Trading Analysis System.

Each scanner is independently testable and handles errors gracefully.

Imports are deferred to avoid failures when individual scanner modules
have dependencies or syntax that are not yet available.
"""


def __getattr__(name):
    """Lazy-load scanner classes on first access."""
    _mapping = {
        "MarketScanner": "scanners.market",
        "TechnicalScanner": "scanners.technical",
        "OptionsScanner": "scanners.options",
        "SentimentScanner": "scanners.sentiment",
        "FlowScanner": "scanners.flow",
        "FinvizScanner": "scanners.finviz",
        "EdgarScanner": "scanners.edgar",
    }
    if name in _mapping:
        import importlib
        mod = importlib.import_module(_mapping[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'scanners' has no attribute {name!r}")


__all__ = [
    "MarketScanner",
    "TechnicalScanner",
    "OptionsScanner",
    "SentimentScanner",
    "FlowScanner",
    "FinvizScanner",
    "EdgarScanner",
]
