from __future__ import annotations

from main_2d import LoopTelemetry


def test_loop_telemetry_accumulates_and_reports() -> None:
    telemetry = LoopTelemetry(started_at=0.0, last_report_at=0.0)
    telemetry.record(1.0, 2.0, 3.0, 6.0)
    telemetry.record(2.0, 4.0, 6.0, 12.0)

    assert telemetry.frames == 2
    assert telemetry.should_report(1.1) is True
