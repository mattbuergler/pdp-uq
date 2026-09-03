from __future__ import annotations

from pdp_uq.cli import main


def test_cli_requires_a_command() -> None:
    try:
        main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should stop when no command is provided")


def test_cli_reports_missing_input(capsys: object) -> None:
    code = main(
        [
            "predict",
            "missing.csv",
            "--dx",
            "0.004",
            "--dy",
            "0.001",
            "--particles-per-window",
            "10",
            "--config",
            "configs/smoke.toml",
        ]
    )
    assert code == 2
