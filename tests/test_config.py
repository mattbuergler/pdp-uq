from __future__ import annotations

from pathlib import Path

import pytest

from pdp_uq.config import load_config
from pdp_uq.exceptions import ConfigurationError


def test_load_smoke_config() -> None:
    config = load_config(Path("configs/smoke.toml"))
    assert config.search.iterations == 2
    assert config.quantiles == (0.05, 0.5, 0.95)
    assert config.model.n_estimators == 3


def test_rejects_unsorted_quantiles(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text("[prediction]\nquantiles = [0.5, 0.1]\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="sorted"):
        load_config(path)
