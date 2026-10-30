"""Configuration-driven thresholds that turn features into diagnostic flags."""
from .config import RuleConfig
from .features import FEATURE_NAMES


def split_feature_key(key: str, channels: tuple[str, ...]) -> tuple[str, str]:
    for feature in sorted(FEATURE_NAMES, key=len, reverse=True):
        suffix = "_" + feature
        if key.endswith(suffix) and key[:-len(suffix)] in channels:
            return key[:-len(suffix)], feature
    raise ValueError(f"rule key must be <channel>_<feature>: {key}")


def apply_rules(feature_values, channels: tuple[str, ...], rules: dict[str, RuleConfig]) -> list[str]:
    flags: list[str] = []
    for key, rule in rules.items():
        channel, feature = split_feature_key(key, channels)
        value = float(feature_values[channels.index(channel), FEATURE_NAMES.index(feature)])
        if rule.maximum is not None and value > rule.maximum:
            flags.append(rule.flag or f"HIGH_{key.upper()}")
        if rule.minimum is not None and value < rule.minimum:
            flags.append(rule.flag or f"LOW_{key.upper()}")
    return flags
