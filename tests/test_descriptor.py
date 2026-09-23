from __future__ import annotations

import unittest

from qfa_agent.descriptor import seal_descriptor, validate_descriptor


class DescriptorTests(unittest.TestCase):
    def test_matches_official_development_fixture_digest(self) -> None:
        descriptor = {
            "schema_version": "1.0.0",
            "interface_version": "2.0",
            "competition_id": "agenthon2026-coding-dev",
            "team_id": "team-example-0001",
            "track": "coding",
            "phase": "dev",
            "category": "api",
            "image": {
                "registry": "docker.io",
                "repository": "team-example/qfb2-solver",
                "digest": "sha256:9586b834b7ab2ce09dbb93d4cc00fff8da6a408f226beb1e1ffb8870165de488",
            },
            "image_access": "public",
            "models": [
                {
                    "name": "example-api-model",
                    "version": "example-model-20260401",
                    "training_cutoff": "2026-01",
                    "access": "api",
                    "revision": "r1",
                }
            ],
            "license": "Apache-2.0",
        }
        sealed = seal_descriptor(descriptor)
        self.assertEqual(
            sealed["descriptor_digest"],
            "sha256:e7e35488525592b766699477a2dcef48e3b87b48bb5de330a71a63e6aa2c210f",
        )
        self.assertEqual(validate_descriptor(sealed), [])

    def test_rejects_retired_byo_category(self) -> None:
        descriptor = {
            "schema_version": "1.0.0",
            "interface_version": "2.0",
            "competition_id": "agenthon2026-coding-dev",
            "team_id": "team-example-0001",
            "track": "coding",
            "phase": "dev",
            "category": "byo-small",
            "image": {
                "registry": "docker.io",
                "repository": "team-example/qfb2-solver",
                "digest": "sha256:" + "0" * 64,
            },
            "image_access": "public",
            "models": [],
            "license": "MIT",
        }
        issues = validate_descriptor(seal_descriptor(descriptor))
        self.assertTrue(any("BYO" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
