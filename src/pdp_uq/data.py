"""Acquisition and validation of immutable external artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import requests

from pdp_uq.exceptions import ArtifactError
from pdp_uq.io import read_json, sha256_file


def _dataset_spec(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    value = manifest.get("dataset")
    if not isinstance(value, dict):
        raise ArtifactError(f"Manifest {manifest_path} has no dataset object.")
    return value


def verify_file(path: Path, expected_sha256: str) -> None:
    """Raise if a file does not match an expected SHA-256 digest."""
    if not path.is_file():
        raise ArtifactError(f"Required artifact does not exist: {path}")
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ArtifactError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, received {actual}."
        )


def _archive_member(archive: zipfile.ZipFile, expected_name: str) -> zipfile.ZipInfo:
    matches: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        normalized = PurePosixPath(member.filename.replace("\\", "/"))
        if normalized.name == expected_name and not member.is_dir():
            matches.append(member)
    if len(matches) != 1:
        raise ArtifactError(
            f"Expected exactly one {expected_name!r} in the dataset archive; found {len(matches)}."
        )
    return matches[0]


def fetch_dataset(
    manifest_path: Path,
    output_path: Path | None = None,
    *,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download, safely extract, and checksum the published training dataset."""
    spec = _dataset_spec(manifest_path)
    destination = output_path or Path(str(spec["output"]))
    expected_hash = str(spec["sha256"])

    if destination.exists():
        verify_file(destination, expected_hash)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    archive_descriptor, archive_name = tempfile.mkstemp(
        prefix=".pdp-uq-dataset.", suffix=".zip", dir=destination.parent
    )
    os.close(archive_descriptor)
    extracted_descriptor, extracted_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(extracted_descriptor)

    archive_path = Path(archive_name)
    extracted_path = Path(extracted_name)
    try:
        with requests.get(
            str(spec["url"]),
            stream=True,
            timeout=(10.0, timeout_seconds),
            headers={"User-Agent": "pdp-uq/1.1"},
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)

        try:
            with zipfile.ZipFile(archive_path) as archive:
                member = _archive_member(archive, str(spec["archive_member"]))
                with archive.open(member) as source, extracted_path.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        except zipfile.BadZipFile as exc:
            raise ArtifactError("The downloaded dataset is not a valid ZIP archive.") from exc

        verify_file(extracted_path, expected_hash)
        os.replace(extracted_path, destination)
        return destination
    except requests.RequestException as exc:
        raise ArtifactError(f"Unable to download the published dataset: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
        extracted_path.unlink(missing_ok=True)


def validate_dataset_artifact(path: Path, manifest_path: Path) -> dict[str, int | str]:
    """Validate checksum, shape, and required training columns."""
    spec = _dataset_spec(manifest_path)
    verify_file(path, str(spec["sha256"]))
    frame = pd.read_csv(path)
    expected_columns = int(spec["columns"])
    expected_rows = int(spec["rows"])
    if frame.shape != (expected_rows, expected_columns):
        raise ArtifactError(
            f"Unexpected dataset shape {frame.shape}; expected "
            f"({expected_rows}, {expected_columns})."
        )
    return {
        "path": str(path),
        "sha256": str(spec["sha256"]),
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
    }
