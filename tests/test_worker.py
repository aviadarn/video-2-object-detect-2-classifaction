from services.pipeline.worker import JobAbortedError, WorkerConfig, deterministic_object_id, merge_outputs, process_job


def test_deterministic_object_id_stable() -> None:
    a = deterministic_object_id("job-1", "7", [1.12349, 2.0, 3.0, 4.0], "car")
    b = deterministic_object_id("job-1", "7", [1.1235, 2.0, 3.0, 4.0], "car")
    assert a == b


def test_merge_outputs_keys() -> None:
    merged = merge_outputs(
        "job-a",
        "f-1",
        detections=[{"box_xyxy": [1, 2, 3, 4], "det_score": 0.9, "det_label": "1"}],
        classifications=[{"swin_class_id": 5, "swin_confidence": 0.6}],
    )
    assert len(merged) == 1
    item = merged[0]
    assert item["job_id"] == "job-a"
    assert item["frame_id"] == "f-1"
    assert item["object_id"]


def test_process_job_aborts_when_failures_exceed_threshold(monkeypatch) -> None:
    manifest = {
        "job_id": "job-err",
        "frames": [
            {"frame_id": "1", "image_path": "missing-1.jpg"},
            {"frame_id": "2", "image_path": "missing-2.jpg"},
        ],
    }
    cfg = WorkerConfig(
        detectron2_url="http://localhost:8000",
        detectron2_model_name="det",
        swin_url="http://localhost:8000",
        swin_model_name="swin",
        max_fail_ratio=0.0,
    )

    try:
        process_job(manifest, cfg)
    except JobAbortedError:
        assert True
    else:
        assert False, "Expected JobAbortedError"
