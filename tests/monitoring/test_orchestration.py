import json

from mibici.monitoring import __main__ as monitoring


def test_all_runs_registered_adapters_and_creates_question_folders(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitoring,
        "run_p1",
        lambda *args: calls.append(args),
    )

    summary = monitoring.run(
        "all",
        "2026-10-01",
        "2026-11-01",
        tmp_path,
        tmp_path / "history.parquet",
        tmp_path / "current.csv",
    )

    assert len(calls) == 1
    assert summary["questions"]["p1"]["status"] == "completed"
    assert summary["questions"]["p1"]["report_dir"] == "p1"
    assert (tmp_path / "p1").is_dir()
    assert (tmp_path / "p2").is_dir()
    assert (tmp_path / "p3").is_dir()
    saved = json.loads((tmp_path / "monitoring_run.json").read_text(encoding="utf-8"))
    assert saved == summary


def test_selecting_question_without_adapter_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "run_p1",
        lambda *args: (_ for _ in ()).throw(AssertionError("P1 must not run")),
    )

    try:
        monitoring.run(
            "p2",
            "2026-10-01",
            "2026-11-01",
            tmp_path,
            tmp_path / "history.parquet",
            tmp_path / "current.csv",
        )
    except ValueError as exc:
        assert "No monitoring adapter is registered for P2" in str(exc)
    else:
        raise AssertionError("Expected unregistered question to fail")


def test_invalid_month_window_is_rejected(tmp_path):
    try:
        monitoring.run(
            "p1",
            "2026-10-02",
            "2026-11-01",
            tmp_path,
            tmp_path / "history.parquet",
            tmp_path / "current.csv",
        )
    except ValueError as exc:
        assert "exactly one calendar month" in str(exc)
    else:
        raise AssertionError("Expected invalid month bounds to fail")
