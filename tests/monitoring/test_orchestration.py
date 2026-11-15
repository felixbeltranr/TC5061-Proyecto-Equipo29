import json

from mibici.monitoring import __main__ as monitoring


def test_all_runs_registered_adapters_and_creates_question_folders(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitoring,
        "run_p1",
        lambda *args: calls.append(args),
    )
    p2_calls = []
    monkeypatch.setattr(
        monitoring, "run_p2", lambda *args, **kwargs: p2_calls.append((args, kwargs))
    )
    p3_calls = []
    monkeypatch.setattr(monitoring, "run_p3", lambda *args, **kwargs: p3_calls.append((args, kwargs)))

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
    assert len(p2_calls) == 1
    assert p2_calls[0][0][2] == tmp_path / "p2"
    assert p2_calls[0][1] == {"nomenclature_path": None}
    assert summary["questions"]["p2"]["report_dir"] == "p2"
    assert len(p3_calls) == 1
    assert p3_calls[0][0][0] == tmp_path / "history.parquet"
    assert p3_calls[0][1] == {"nomenclature_path": None}
    assert summary["questions"]["p3"]["report_dir"] == "p3"
    assert (tmp_path / "p1").is_dir()
    assert (tmp_path / "p2").is_dir()
    assert (tmp_path / "p3").is_dir()
    saved = json.loads((tmp_path / "monitoring_run.json").read_text(encoding="utf-8"))
    assert saved == summary


def test_p2_runs_only_its_adapter(tmp_path, monkeypatch):
    def no_debe_correr(*args, **kwargs):
        raise AssertionError("Only P2 must run")

    monkeypatch.setattr(monitoring, "run_p1", no_debe_correr)
    monkeypatch.setattr(monitoring, "run_p3", no_debe_correr)
    p2_calls = []
    monkeypatch.setattr(
        monitoring, "run_p2", lambda *args, **kwargs: p2_calls.append((args, kwargs))
    )

    summary = monitoring.run(
        "p2",
        "2026-10-01",
        "2026-11-01",
        tmp_path,
        tmp_path / "history.parquet",
        tmp_path / "current.csv",
        nomenclature=tmp_path / "nomenclatura.csv",
    )

    assert len(p2_calls) == 1
    assert p2_calls[0][1] == {"nomenclature_path": tmp_path / "nomenclatura.csv"}
    assert list(summary["questions"]) == ["p2"]


def test_unknown_question_fails_explicitly(tmp_path):
    try:
        monitoring.run(
            "p4",
            "2026-10-01",
            "2026-11-01",
            tmp_path,
            tmp_path / "history.parquet",
            tmp_path / "current.csv",
        )
    except ValueError as exc:
        assert "question must be one of" in str(exc)
    else:
        raise AssertionError("Expected unknown question to fail")


def test_p3_runs_only_its_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(
        monitoring,
        "run_p1",
        lambda *args: (_ for _ in ()).throw(AssertionError("P1 must not run")),
    )
    p3_calls = []
    monkeypatch.setattr(monitoring, "run_p3", lambda *args, **kwargs: p3_calls.append((args, kwargs)))

    summary = monitoring.run(
        "p3",
        "2026-10-01",
        "2026-11-01",
        tmp_path,
        tmp_path / "history.parquet",
        tmp_path / "current.csv",
    )

    assert len(p3_calls) == 1
    assert list(summary["questions"]) == ["p3"]


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
