import json
import logging

from soccer_edge.app_logging import JsonFormatter, configure_logging, get_logger, log_event


def test_json_formatter_includes_extra_fields() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.extra_fields = {"match_id": "m1"}
    payload = json.loads(formatter.format(record))
    assert payload["message"] == "hello"
    assert payload["match_id"] == "m1"


def test_configure_logging_and_log_event() -> None:
    configure_logging(json_logs=True)
    logger = get_logger("soccer_edge.test")
    log_event(logger, "event", match_id="m1")
    assert logging.getLogger().handlers
