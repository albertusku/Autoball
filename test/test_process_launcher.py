"""Integration tests for the process launching utilities."""
import io
import logging
import subprocess
import sys
import unittest

from auto_utils import launch_process


class LaunchProcessIntegrationTests(unittest.TestCase):
    """Verify that processes with large stdout/stderr output do not block."""

    def setUp(self) -> None:
        logger_name = f"{self.__class__.__name__}.{self._testMethodName}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.stream = io.StringIO()
        handler = logging.StreamHandler(self.stream)
        handler.setLevel(logging.DEBUG)
        for existing in list(self.logger.handlers):
            self.logger.removeHandler(existing)
        self.logger.addHandler(handler)
        self._handler = handler

    def tearDown(self) -> None:
        self.logger.removeHandler(self._handler)
        self._handler.close()

    def test_large_output_process_completes(self) -> None:
        chunk_size = 70_000  # >64 KB ensures the pipe buffers would fill without draining
        script = (
            "import sys\n"
            f"sys.stdout.write('A' * {chunk_size})\n"
            "sys.stdout.flush()\n"
            f"sys.stderr.write('B' * {chunk_size})\n"
            "sys.stderr.flush()\n"
        )

        process = launch_process(
            [sys.executable, "-c", script],
            logger=self.logger,
            name="dummy-large-output",
        )

        try:
            exit_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            self.fail(f"Process blocked due to undrained pipes: {exc}")

        self.assertEqual(exit_code, 0)
        logged_output = self.stream.getvalue()
        self.assertIn("dummy-large-output stdout", logged_output)
        self.assertIn("dummy-large-output stderr", logged_output)


if __name__ == "__main__":
    unittest.main()
