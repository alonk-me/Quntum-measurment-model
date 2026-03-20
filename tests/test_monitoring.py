"""Tests for the monitoring module (logging utilities)."""

import logging
import time

import pytest

from quantum_measurement.monitoring import configure_logging, ExperimentLogger


class TestConfigureLogging:
    def test_returns_logger(self):
        logger = configure_logging(level="WARNING")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = configure_logging(logger_name="test_qm_monitor")
        assert logger.name == "test_qm_monitor"

    def test_level_set(self):
        logger = configure_logging(level="DEBUG", logger_name="test_qm_debug")
        assert logger.level == logging.DEBUG

    def test_file_handler_created(self, tmp_path):
        log_file = str(tmp_path / "subdir" / "test.log")
        configure_logging(level="INFO", log_file=log_file, logger_name="test_qm_file")
        import os
        assert os.path.exists(log_file)

    def test_idempotent(self):
        """Calling configure_logging twice should not add duplicate handlers."""
        name = "test_qm_idempotent"
        logger1 = configure_logging(level="INFO", logger_name=name)
        n_handlers_1 = len(logger1.handlers)
        logger2 = configure_logging(level="INFO", logger_name=name)
        assert len(logger2.handlers) == n_handlers_1


class TestExperimentLogger:
    def test_context_manager_no_exception(self):
        with ExperimentLogger("unit test") as el:
            pass
        assert el.elapsed >= 0

    def test_elapsed_is_positive(self):
        with ExperimentLogger("timing test") as el:
            time.sleep(0.01)
        assert el.elapsed >= 0.01

    def test_does_not_suppress_exceptions(self):
        with pytest.raises(ValueError, match="boom"):
            with ExperimentLogger("error test"):
                raise ValueError("boom")

    def test_custom_logger(self):
        custom = logging.getLogger("test_qm_custom_el")
        with ExperimentLogger("custom logger test", logger=custom):
            pass
