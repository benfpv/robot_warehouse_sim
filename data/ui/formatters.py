"""Pure display formatting helpers.

Extracted from ``main.MainGame`` so they can be unit-tested in isolation.
All functions are deterministic and free of side effects.
"""
from __future__ import annotations


def fmt_elapsed(s: float) -> str:
    """Format seconds as a concise two-unit human-readable duration.

    Examples
    --------
    >>> fmt_elapsed(0)
    '0s'
    >>> fmt_elapsed(75)
    '1min 15s'
    >>> fmt_elapsed(3600)
    '1hr'
    """
    if s < 0:
        return "0s"
    intervals = [
        (365.25 * 24 * 3600 * 10, "decade", "decades"),
        (365.25 * 24 * 3600,       "yr",     "yrs"),
        (30.44  * 24 * 3600,       "mo",     "mo"),
        (24 * 3600,                "d",      "d"),
        (3600,                     "hr",     "hrs"),
        (60,                       "min",    "mins"),
        (1,                        "s",      "s"),
    ]
    parts: list[str] = []
    rem = float(s)
    for secs, sg, pl in intervals:
        if rem >= secs:
            n = int(rem // secs)
            rem -= n * secs
            parts.append("{}{}".format(n, pl if n != 1 else sg))
            if len(parts) == 2:
                break
    return " ".join(parts) if parts else "0s"


def fmt_age(seconds: float) -> str:
    """Short age string: e.g. ``'3s'``, ``'2m12s'``, ``'1h5m'``."""
    s = int(seconds)
    if s < 60:
        return "{}s".format(s)
    if s < 3600:
        return "{}m{}s".format(s // 60, s % 60)
    if s < 86400:
        return "{}h{}m".format(s // 3600, (s % 3600) // 60)
    return "{}d{}h".format(s // 86400, (s % 86400) // 3600)


def fmt_rate(val: float) -> str:
    """Format a per-second rate, abbreviating large magnitudes."""
    if val == 0:
        return "0/s"
    a = abs(val)
    if a >= 1_000_000:
        return "{:+.1f}M/s".format(val / 1_000_000)
    if a >= 10_000:
        return "{:+.1f}K/s".format(val / 1_000)
    if a >= 10:
        return "{:+.0f}/s".format(val)
    return "{:+.1f}/s".format(val)


def fmt_n(val: int) -> str:
    """Abbreviate large integers for display (``1500000`` -> ``'1.5M'``)."""
    if val >= 1_000_000:
        return "{:.1f}M".format(val / 1_000_000)
    if val >= 10_000:
        return "{:.1f}K".format(val / 1_000)
    return str(val)
