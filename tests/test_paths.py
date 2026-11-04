from datetime import datetime, timezone

from bt.paths import RunOutputPaths, latest_run_dir, make_timestamped_run_dir


def test_make_timestamped_run_dir_uses_utc_stamp(tmp_path):
    when = datetime(2026, 5, 28, 20, 35, 0, tzinfo=timezone.utc)
    run_dir = make_timestamped_run_dir(tmp_path, run_at=when)
    assert run_dir == tmp_path / "20260528_203500"
    assert run_dir.is_dir()


def test_run_output_paths_names(tmp_path):
    when = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    paths = RunOutputPaths(run_dir=make_timestamped_run_dir(tmp_path, run_at=when))
    assert paths.trades.name == "trades.csv"
    assert paths.metadata.name == "metadata.json"
    assert len(paths.artifact_paths()) == 5


def test_latest_run_dir_includes_walkforward_folders(tmp_path):
    (tmp_path / "20260805_080551").mkdir()
    (tmp_path / "sweep_20260805_080557").mkdir()
    wf = tmp_path / "walkforward_20260805_081000"
    wf.mkdir()
    assert latest_run_dir(tmp_path) == wf
