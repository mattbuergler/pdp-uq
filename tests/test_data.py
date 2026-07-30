from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import responses

from pdp_uq.data import (
    _archive_member,
    fetch_dataset,
    validate_dataset_artifact,
    verify_file,
)
from pdp_uq.exceptions import ArtifactError
from pdp_uq.io import sha256_file, write_json


def test_sha256_and_atomic_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"pdp-uq")
    digest = sha256_file(artifact)
    verify_file(artifact, digest)
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        verify_file(artifact, "0" * 64)

    output = tmp_path / "nested" / "record.json"
    write_json(output, {"digest": digest})
    assert json.loads(output.read_text(encoding="utf-8")) == {"digest": digest}


def test_archive_member_matches_by_safe_basename(tmp_path: Path) -> None:
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/simulation_results.csv", "a,b\n1,2\n")
        archive.writestr("../../irrelevant.txt", "ignored")
    with zipfile.ZipFile(path) as archive:
        member = _archive_member(archive, "simulation_results.csv")
    assert member.filename == "nested/simulation_results.csv"


@responses.activate
def test_fetches_extracts_and_validates_dataset(tmp_path: Path) -> None:
    csv_bytes = b"a,b\n1,2\n"
    archive_bytes = BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("published/simulation_results.csv", csv_bytes)

    url = "https://example.test/dataset.zip"
    responses.add(responses.GET, url, body=archive_bytes.getvalue(), status=200)
    digest_path = tmp_path / "expected.csv"
    digest_path.write_bytes(csv_bytes)
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "data" / "simulation_results.csv"
    write_json(
        manifest,
        {
            "dataset": {
                "url": url,
                "archive_member": "simulation_results.csv",
                "output": str(output),
                "sha256": sha256_file(digest_path),
                "rows": 1,
                "columns": 2,
            }
        },
    )

    assert fetch_dataset(manifest) == output
    assert pd.read_csv(output).to_dict(orient="records") == [{"a": 1, "b": 2}]
    assert fetch_dataset(manifest) == output
    report = validate_dataset_artifact(output, manifest)
    assert report["rows"] == 1
    assert report["columns"] == 2
