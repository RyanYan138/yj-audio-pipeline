#!/usr/bin/env python3

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ShellLauncherTest(unittest.TestCase):
    def test_user_launchers_fall_back_from_an_incomplete_conda_environment(self):
        launchers = [
            "run_funasr_nano_batch_server_tuned.sh",
            "run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh",
        ]

        for launcher in launchers:
            with self.subTest(launcher=launcher):
                script = (PROJECT_ROOT / launcher).read_text(encoding="utf-8")
                self.assertIn('DEFAULT_ENV_PREFIX=', script)
                self.assertIn('FUNASR_PYTHON', script)
                self.assertIn('RUNTIME_PREFIX=', script)
                self.assertIn('export CONDA_PREFIX="${RUNTIME_PREFIX}"', script)

    def test_metadata_mode_rejects_python_without_lid_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prefix = Path(temp_dir)
            fake_python = prefix / "bin/python"
            fake_python.parent.mkdir()
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ ${2:-} == *ctranslate2* ]]; then exit 1; fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = {
                **os.environ,
                "FUNASR_PYTHON": str(fake_python),
                "LID_MODE": "metadata",
            }

            result = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh"),
                ],
                capture_output=True,
                check=False,
                env=environment,
                text=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing FunASR runtime dependencies", result.stdout)

    def test_perf_sweep_expands_empty_timestamp_array_under_nounset(self):
        script = (PROJECT_ROOT / "test/run_nano_auto_lang_perf_sweep.sh").read_text()

        self.assertIn(
            '${timestamp_args[@]+"${timestamp_args[@]}"}',
            script,
        )

    def test_pipeline_launcher_expands_empty_extra_args_under_nounset(self):
        script = (
            PROJECT_ROOT / "run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}', script)

    def test_perf_sweep_can_resume_selected_configurations(self):
        script = (PROJECT_ROOT / "test/run_nano_auto_lang_perf_sweep.sh").read_text()

        self.assertIn('CONFIGS="${CONFIGS:-', script)
        self.assertIn('for config in ${CONFIGS}; do', script)

    def test_pipeline_launcher_defaults_to_measured_fast_profile(self):
        script = (
            PROJECT_ROOT / "run_tar_pipeline_nodnsmos_nano_auto_lang_ckpt.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('ASR_BATCH="${4:-96}"', script)
        self.assertIn('VAD_WORKERS="${VAD_WORKERS:-4}"', script)
        self.assertIn('LID_MODE="${LID_MODE:-off}"', script)

    def test_codeswitch_ab_explicitly_enables_lid_metadata(self):
        script = (PROJECT_ROOT / "test/run_codeswitch_ab.sh").read_text()

        self.assertIn("LID_MODE=metadata", script)


if __name__ == "__main__":
    unittest.main()
