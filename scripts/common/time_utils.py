from __future__ import annotations

from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def ts_local(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)


def ts_utc(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now(timezone.utc).strftime(fmt)


def ts_shanghai(fmt: str = "%Y%m%d_%H%M%S") -> str:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Asia/Shanghai")).strftime(fmt)
        except Exception:
            pass
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime(fmt)

