from pathlib import Path
from shuttersift.engine.state import StateManager
from shuttersift.engine import PhotoResult, SubScores


def _make_result(name: str, score: float = 75.0) -> PhotoResult:
    return PhotoResult(
        path=Path(f"/photos/{name}.jpg"),
        score=score,
        decision="keep",
    )


def test_empty_state_has_no_processed(tmp_path):
    sm = StateManager(tmp_path)
    assert not sm.is_processed(Path("/photos/test.jpg"))


def test_save_and_resume(tmp_path):
    sm = StateManager(tmp_path)
    result = _make_result("DSC001", score=82.5)
    sm.save(result)

    sm2 = StateManager(tmp_path)
    assert sm2.is_processed(result.path)
    loaded = sm2.load(result.path)
    assert loaded is not None
    assert abs(loaded.score - 82.5) < 0.01
    assert loaded.decision == "keep"


def test_fresh_ignores_state(tmp_path):
    sm = StateManager(tmp_path)
    result = _make_result("DSC002")
    sm.save(result)

    sm_fresh = StateManager(tmp_path, fresh=True)
    assert not sm_fresh.is_processed(result.path)


def test_state_file_written_to_output_dir(tmp_path):
    sm = StateManager(tmp_path)
    sm.save(_make_result("DSC003"))
    assert (tmp_path / ".state.json").exists()
