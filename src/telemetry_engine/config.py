"""Validated YAML configuration for alignment, batching, and rules."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math

import yaml


@dataclass(frozen=True)
class RuleConfig:
    minimum: float | None = None
    maximum: float | None = None
    flag: str | None = None

    def __post_init__(self) -> None:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a diagnostic rule needs min or max")
        for value in (self.minimum, self.maximum):
            if value is not None and not math.isfinite(value):
                raise ValueError("diagnostic thresholds must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("diagnostic rule min cannot exceed max")
        if self.flag is not None and not self.flag.strip():
            raise ValueError("diagnostic flag cannot be empty")


@dataclass(frozen=True)
class EngineConfig:
    target_rate_hz: float = 10.0
    alignment: str = "linear"
    batch_size: int = 64
    required_channels: tuple[str, ...] = ()
    value_ranges: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    rules: dict[str, RuleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_rate_hz) or self.target_rate_hz <= 0:
            raise ValueError("target_rate_hz must be finite and positive")
        if self.alignment not in {"linear", "nearest"}:
            raise ValueError("alignment must be 'linear' or 'nearest'")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if len(set(self.required_channels)) != len(self.required_channels):
            raise ValueError("required_channels cannot contain duplicates")
        for name, limits in self.value_ranges.items():
            if not name or len(limits) != 2:
                raise ValueError("value_ranges entries require [min, max]")
            low, high = limits
            if any(value is not None and not math.isfinite(value) for value in limits):
                raise ValueError("value range bounds must be finite")
            if low is not None and high is not None and low > high:
                raise ValueError(f"invalid value range for {name}")


def _strict_keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(sorted(unknown))}")


def load_config(path: str | Path | None) -> EngineConfig:
    if path is None:
        return EngineConfig()
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a mapping")
    _strict_keys(raw, {"target_rate_hz", "alignment", "batch_size", "required_channels", "value_ranges", "rules"}, "configuration")
    rules: dict[str, RuleConfig] = {}
    for feature, value in raw.get("rules", {}).items():
        if not isinstance(value, dict):
            raise ValueError(f"rule {feature} must be a mapping")
        _strict_keys(value, {"min", "max", "flag"}, f"rule {feature}")
        rules[feature] = RuleConfig(value.get("min"), value.get("max"), value.get("flag"))
    ranges = {}
    for channel, value in raw.get("value_ranges", {}).items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"value range for {channel} must be [min, max]")
        ranges[channel] = (value[0], value[1])
    return EngineConfig(
        target_rate_hz=raw.get("target_rate_hz", 10.0),
        alignment=raw.get("alignment", "linear"),
        batch_size=raw.get("batch_size", 64),
        required_channels=tuple(raw.get("required_channels", ())),
        value_ranges=ranges,
        rules=rules,
    )

