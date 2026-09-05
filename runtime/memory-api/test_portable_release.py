from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from ops.build_tmcra_service_release import build_release


ROOT = Path(__file__).resolve().parent


class PortableReleaseTests(unittest.TestCase):
    def test_installer_keeps_huggingface_hub_compatible_with_transformers(self) -> None:
        installer = (ROOT / "deploy" / "install-tmcra.sh").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements-tmcra-service.txt").read_text(encoding="utf-8")

        self.assertIn("transformers>=4.45,<5", requirements)
        self.assertIn(
            'pip install --upgrade "huggingface_hub>=0.34,<1.0"',
            installer,
        )
        self.assertNotIn("pip install --upgrade huggingface_hub\n", installer)

    def test_installer_downloads_only_runtime_model_files(self) -> None:
        installer = (ROOT / "deploy" / "install-tmcra.sh").read_text(encoding="utf-8")

        self.assertIn('HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"', installer)
        self.assertIn('-f "$EMBEDDING_MODEL/pytorch_model.bin"', installer)
        self.assertIn('-f "$CROSS_MODEL/model.safetensors"', installer)
        self.assertIn("downloaded BGE-M3 artifact failed SHA-256 verification", installer)
        self.assertIn(
            "downloaded BGE reranker artifact failed SHA-256 verification",
            installer,
        )
        self.assertIn("bge-reranker-v2-m3.TMCRA_MODEL_MANIFEST.json", installer)
        self.assertIn("TMCRA_INTEGRATED_REPO=$PREFIX", installer)
        self.assertNotIn("TMCRA_INTEGRATED_REPO=$DATA_DIR/repository", installer)
        self.assertIn("1_Pooling/config.json colbert_linear.pt config.json", installer)
        self.assertIn("config.json model.safetensors sentencepiece.bpe.model", installer)
        self.assertNotIn(
            'download_model "$BGE_REPO" "$BGE_REVISION" "$EMBEDDING_MODEL"\n',
            installer,
        )
        self.assertNotIn(
            'download_model "$CROSS_REPO" "$CROSS_REVISION" "$CROSS_MODEL"\n',
            installer,
        )

    def test_controls_resolve_runtime_paths_after_loading_service_env(self) -> None:
        control = (ROOT / "deploy" / "tmcra-memory-api-control.sh").read_text(
            encoding="utf-8"
        )
        maintenance = (ROOT / "deploy" / "tmcra-production-maintenance.sh").read_text(
            encoding="utf-8"
        )

        control_source = control.index('source "$ENV_FILE"')
        self.assertGreater(
            control.index('PYTHON="${TMCRA_SERVICE_PYTHON:', control_source),
            control_source,
        )
        maintenance_source = maintenance.index('source "$ENV_FILE"')
        for assignment in (
            'API_CONTROL="${TMCRA_MEMORY_API_CONTROL:',
            'LOCAL_LLM_CONTROL="${TMCRA_LOCAL_LLM_CONTROL:',
            'PYTHON="${TMCRA_SERVICE_PYTHON:',
        ):
            self.assertGreater(
                maintenance.index(assignment, maintenance_source),
                maintenance_source,
            )

    def test_preflight_script_is_directly_executable(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "ops" / "run_tmcra_service_preflight.py"),
                "--help",
            ],
            cwd=ROOT.parent,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--env-file", result.stdout)

    def test_runtime_dependency_closure_is_in_the_service_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            archive = work / "service.tar.gz"
            extract = work / "extract"
            build_release(ROOT, archive)
            with tarfile.open(archive, "r:gz") as handle:
                names = set(handle.getnames())
                handle.extractall(extract)

            self.assertIn("experiments/replacement/adapters/memory_adapters.py", names)
            self.assertIn("experiments/replacement/memory_graph.py", names)
            self.assertIn("core/session_memory.py", names)
            self.assertIn("build_v3_runtime_dataset.py", names)
            self.assertIn("tmcra_v3_reranker.py", names)
            self.assertIn("tmcra_v3_schema.py", names)
            self.assertIn("models/tmcra_v3_reranker.pt", names)
            self.assertIn("deploy/install-tmcra.sh", names)
            self.assertIn("deploy/tmcra", names)
            self.assertIn("deploy/tmcra-local-llm-control.sh", names)
            self.assertIn("deploy/tmcra-production-maintenance.sh", names)
            self.assertIn(
                "deploy/model-manifests/bge-reranker-v2-m3.TMCRA_MODEL_MANIFEST.json",
                names,
            )
            self.assertIn("deploy/writer.env.example", names)
            self.assertIn("ops/run_commercial_api_smoke.py", names)
            self.assertNotIn("ops/run_launch_api_smoke.py", names)

            writer_template = (extract / "deploy" / "writer.env.example").read_bytes()
            self.assertNotIn(b"\r\n", writer_template)
            self.assertIn(b"TMCRA_WRITER_PROVIDER=local-qwen\n", writer_template)

            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONPATH"] = str(extract)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from experiments.replacement.adapters.memory_adapters "
                    "import GraphSessionMemoryAdapter; "
                    "from core.session_memory import SessionMemoryExtractor; "
                    "print('portable-import-ok')",
                ],
                cwd=extract,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("portable-import-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
