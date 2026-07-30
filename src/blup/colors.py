from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping


DEFAULT_BASE_COLORS: tuple[str, ...] = (
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
)

DEFAULT_OTHER_COLOR = "#b0b0b0"
DEFAULT_FALLBACK_COLOR = "#999999"


@dataclass(frozen=True)
class TokenColorSnapshot:
    color_map: Mapping[str, str]

    def get_color(self, token_key: str) -> str:
        return self.color_map.get(token_key, DEFAULT_FALLBACK_COLOR)


@dataclass
class TokenColor:
    base_colors: tuple[str, ...] = DEFAULT_BASE_COLORS
    other_color: str = DEFAULT_OTHER_COLOR
    fallback_color: str = DEFAULT_FALLBACK_COLOR

    _known_keys: set[str] = field(default_factory=set)
    _overrides: dict[str, str] = field(default_factory=dict)

    def register_tokens(
        self,
        token_keys: Iterable[str],
        *,
        token_names: Mapping[str, str] | None = None,
        namespace: str | None = None,
    ) -> None:
        # token_names/namespace not currently implemented
        del token_names
        del namespace

        self._known_keys.update(str(k) for k in token_keys)

    def set_color_override(self, token_key: str, color: str) -> None:
        self._overrides[str(token_key)] = color

    def clear_color_override(self, token_key: str) -> None:
        self._overrides.pop(str(token_key), None)

    def get_color(
        self,
        token_key: str,
        *,
        token_name: str | None = None,
    ) -> str:
        # token_name not currently implemented
        del token_name

        token_key = str(token_key)

        if token_key == "OTHER":
            return self.other_color
        if token_key in self._overrides:
            return self._overrides[token_key]
        return self._generated_color(token_key)

    def snapshot(
        self,
        token_keys: Iterable[str],
        *,
        token_names: Mapping[str, str] | None = None,
    ) -> TokenColorSnapshot:
        # token_names not currently implemented
        del token_names

        keys = {str(k) for k in token_keys}
        keys.add("OTHER")
        self._known_keys.update(keys)

        color_by_key: dict[str, str] = {}

        ordered_keys = sorted(k for k in keys if k != "OTHER")

        for i, key in enumerate(ordered_keys):
            if key in self._overrides:
                color_by_key[key] = self._overrides[key]
            elif i < len(self.base_colors):
                color_by_key[key] = self.base_colors[i]
            else:
                color_by_key[key] = self._generated_color(key)

        color_by_key["OTHER"] = self.other_color
        return TokenColorSnapshot(color_map=color_by_key)

    def snapshot_registered(self) -> TokenColorSnapshot:
        return self.snapshot(self._known_keys)

    def _generated_color(self, key: str) -> str:
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        hue = int.from_bytes(h[:2], "big") / 65535.0
        sat = 0.55 + (h[2] / 255.0) * 0.25
        val = 0.65 + (h[3] / 255.0) * 0.25
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
