"""Seal and minimally validate an Agenthon submission descriptor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


_REQUIRED = {
    "schema_version",
    "interface_version",
    "competition_id",
    "team_id",
    "track",
    "phase",
    "category",
    "image",
    "image_access",
    "models",
    "license",
    "descriptor_digest",
}
_SHA_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_payload(descriptor: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in descriptor.items() if key != "descriptor_digest"}
    # Descriptor values contain no non-integral numbers; compact sorted JSON is
    # therefore byte-identical to the official fixture's canonical form.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def seal_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(descriptor)
    sealed.pop("descriptor_digest", None)
    digest = hashlib.sha256(canonical_payload(sealed)).hexdigest()
    sealed["descriptor_digest"] = f"sha256:{digest}"
    return sealed


def validate_descriptor(descriptor: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    keys = set(descriptor)
    if keys != _REQUIRED:
        missing = sorted(_REQUIRED - keys)
        unknown = sorted(keys - _REQUIRED)
        if missing:
            issues.append(f"missing fields: {missing}")
        if unknown:
            issues.append(f"unknown fields: {unknown}")
    if descriptor.get("schema_version") != "1.0.0":
        issues.append('schema_version must be "1.0.0"')
    if descriptor.get("interface_version") != "2.0":
        issues.append('interface_version must be "2.0"')
    if descriptor.get("track") != "coding":
        issues.append('track must be "coding"')
    if descriptor.get("phase") not in {"dev", "final", "verification"}:
        issues.append("invalid phase")
    if descriptor.get("category") != "api":
        issues.append('T1 category must be "api"; BYO models/adapters are no longer allowed')
    if descriptor.get("image_access") not in {"public", "organizer_mirror"}:
        issues.append("invalid image_access")
    image = descriptor.get("image")
    if not isinstance(image, dict) or set(image) != {"registry", "repository", "digest"}:
        issues.append("image must contain exactly registry, repository, and digest")
    elif not isinstance(image.get("digest"), str) or not _SHA_RE.fullmatch(image["digest"]):
        issues.append("image.digest must be sha256 plus 64 lowercase hex characters")
    models = descriptor.get("models")
    if not isinstance(models, list):
        issues.append("models must be an array")
    else:
        expected = {"name", "version", "training_cutoff", "access", "revision"}
        for index, model in enumerate(models):
            if not isinstance(model, dict) or set(model) != expected:
                issues.append(f"models[{index}] has wrong fields")
            elif model.get("access") != "api":
                issues.append(f'models[{index}].access must be "api"')
    digest = descriptor.get("descriptor_digest")
    if isinstance(digest, str) and _SHA_RE.fullmatch(digest):
        expected_digest = seal_descriptor(descriptor)["descriptor_digest"]
        if digest != expected_digest:
            issues.append("descriptor_digest does not match canonical descriptor content")
    else:
        issues.append("descriptor_digest is malformed")
    return issues


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seal an Agenthon submission.json descriptor")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    descriptor = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(descriptor, dict):
        parser.error("descriptor must be a JSON object")
    if args.check:
        issues = validate_descriptor(descriptor)
        if issues:
            for issue in issues:
                print(issue)
            return 1
        print("descriptor is valid")
        return 0
    sealed = seal_descriptor(descriptor)
    output = args.out or args.input
    _atomic_json(output, sealed)
    issues = validate_descriptor(sealed)
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(sealed["descriptor_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
