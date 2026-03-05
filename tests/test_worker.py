from pathlib import Path

from services.pipeline.worker import (
    PipelineConfig,
    PipelineRunner,
    SampledFrame,
    SimpleObjectTracker,
    build_report,
)


class FakeSceneSplitter:
    def split(self, video_path: Path) -> list[SampledFrame]:
        return [
            SampledFrame(scene_id="s1", frame_role="first", frame_index=0, timestamp_s=0.0, image_path=Path("a.jpg")),
            SampledFrame(scene_id="s1", frame_role="middle", frame_index=1, timestamp_s=0.5, image_path=Path("b.jpg")),
            SampledFrame(scene_id="s1", frame_role="last", frame_index=2, timestamp_s=1.0, image_path=Path("c.jpg")),
        ]


class FakeDetector:
    name = "fake"

    def detect(self, frame: SampledFrame):
        return [{"box_xyxy": [0, 0, 10, 10], "score": 0.9, "label": "person"}]


class FakeClassifier:
    def classify(self, frame: SampledFrame, detection: dict):
        return {"label": "walking", "confidence": 0.88}


def test_pipeline_runner_produces_report_shape(tmp_path: Path) -> None:
    runner = PipelineRunner(
        config=PipelineConfig(job_id="job-1", video_path=tmp_path / "video.mp4"),
        scene_splitter=FakeSceneSplitter(),
        detectors=[FakeDetector()],
        classifier=FakeClassifier(),
        tracker=SimpleObjectTracker(max_distance=50.0),
    )

    report = runner.run()

    assert report["job_id"] == "job-1"
    assert report["pipeline"] == [
        "split",
        "object_detection",
        "classification",
        "object_tracking",
        "clustering",
        "report",
    ]
    assert len(report["frames"]) == 3
    assert report["summary"]["total_detections"] == 3
    assert report["summary"]["total_tracks"] == 1


def test_build_report_clusters_by_classification() -> None:
    records = [
        {"classification": {"label": "cat"}, "track_id": "t1"},
        {"classification": {"label": "cat"}, "track_id": "t2"},
        {"classification": {"label": "dog"}, "track_id": "t3"},
    ]

    clustered = build_report(job_id="job-a", frame_records=[], detections=records)

    clusters = {item["cluster_key"]: item["track_ids"] for item in clustered["clusters"]}
    assert clusters["cat"] == ["t1", "t2"]
    assert clusters["dog"] == ["t3"]
