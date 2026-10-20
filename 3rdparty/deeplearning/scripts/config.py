"""Python config system for FlashOCC-torch (mmengine-free).

Config files are plain .py modules under configs/: module-level variables are
collected into a Config (dict with attribute access). Shared values live in
configs/common.py and datasets/definitions.py; leaf configs import them
directly — no ${} interpolation, no directory-inheritance merging.

CLI overrides keep the torchpack style: ['--model.fuser.out_channels=128']
or ['a.b', 'value'].
"""

import json
from ast import literal_eval
from types import FunctionType, ModuleType
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

__all__ = ["Config", "load_config"]

# module-level names of these types are helpers/imports, not config entries
SKIP_TYPES = (ModuleType, FunctionType, type)


def to_config(obj):
    """Recursively convert dicts to Config, tuples to lists (dump-safe)."""
    if isinstance(obj, dict):
        return Config({k: to_config(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [to_config(v) for v in obj]
    return obj


def plain(obj):
    """Recursively convert Config back to plain dict/list (tuple -> list)."""
    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    return obj


class Config(dict):
    """dict with attribute access (torchpack-compatible)."""

    def __getattr__(self, key: str) -> Any:
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    def get(self, key, default=None):
        return dict.get(self, key, default)

    def update(self, other: Optional[Dict] = None, **kwargs) -> None:  # type: ignore[override]
        """Deep-merge dicts (torchpack semantics): nested dicts merge
        recursively, everything else replaces."""
        if other is None:
            other = {}
        for key, value in list(other.items()) + list(kwargs.items()):
            if isinstance(value, dict):
                if key not in self or not isinstance(self[key], Config):
                    self[key] = Config()
                self[key].update(value)
            else:
                self[key] = value

    def update_from_options(self, opts: Union[List[str], Tuple[str, ...]]) -> None:
        """Apply torchpack-style CLI overrides.

        Accepts ['--a.b=value', ...] or ['a.b', 'value', ...].
        """
        index = 0
        while index < len(opts):
            opt = opts[index]
            if opt.startswith("--"):
                opt = opt[2:]
            if "=" in opt:
                key, value = opt.split("=", 1)
                index += 1
            else:
                key, value = opt, opts[index + 1]
                index += 2
            current = self
            subkeys = key.split(".")
            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            for subkey in subkeys[:-1]:
                current = current.setdefault(subkey, Config())
            current[subkeys[-1]] = value

    def dict(self) -> Dict[str, Any]:
        return plain(self)

    def hash(self) -> str:
        import hashlib

        buffer = json.dumps(self.dict(), sort_keys=True, default=str)
        return hashlib.sha256(buffer.encode()).hexdigest()

    def dump(self, fpath: str) -> None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
        with open(fpath, "w") as f:
            yaml.safe_dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def __str__(self) -> str:
        texts = []
        for key, value in self.items():
            if isinstance(value, Config):
                seperator = "\n"
            else:
                seperator = " "
            text = key + ":" + seperator + str(value)
            lines = text.split("\n")
            for k, line in enumerate(lines[1:]):
                lines[k + 1] = (" " * 2) + line
            texts.extend(lines)
        return "\n".join(texts)


def load_config(fpath: str, options: Optional[List[str]] = None) -> Config:
    """Exec a .py config file and collect its module-level data variables.

    Modules/functions/classes in the namespace are skipped (they are imports
    or builder helpers); nested dicts become attribute-accessible Config.
    CLI `options` are applied to the final config.
    """
    import os
    import sys

    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    with open(fpath, "r") as f:
        source = f.read()
    ns = {"__file__": os.path.abspath(fpath), "__name__": "__config__"}
    exec(compile(source, fpath, "exec"), ns)

    cfg = Config()
    for key, value in ns.items():
        if key.startswith("__") or isinstance(value, SKIP_TYPES):
            continue
        cfg[key] = to_config(value)
    if options:
        cfg.update_from_options(options)
    return cfg
