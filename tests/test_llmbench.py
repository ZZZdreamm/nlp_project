import json
import random

from llmbench import bench, cli, gpu
from llmbench.backends import FakeBackend

CFG = cli.load_config(None)


def test_slug():
    g = gpu.GPUInfo(name="NVIDIA A100-SXM4-40GB", memory_total_gb=40.0)
    assert g.slug == "a100-sxm4-40gb"
    assert gpu.GPUInfo(name="NVIDIA GeForce RTX 4090", memory_total_gb=24).slug == "rtx-4090-24gb"


def test_detect_never_crashes():
    assert isinstance(gpu.detect_gpu(0), gpu.GPUInfo)


def test_select_models():
    ids = lambda v: [m["id"].split("/")[-1] for m in bench.select_models(CFG["models"], v, 0.7)]
    assert ids(8) == ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"]
    assert "Qwen2.5-7B-Instruct" in ids(24) and "Qwen2.5-32B-Instruct" not in ids(24)
    assert "Qwen2.5-14B-Instruct" in ids(80)


def test_run_model_fake_finds_oom():
    be = FakeBackend()
    res = bench.run_model(be, {"id": "x/fake", "params_b": 1}, CFG, random.Random(0), log=lambda *_: None)
    assert res["status"] == "ok" and res["latency"]["decode_tok_s"] > 0
    assert res["max_batch_ok"] >= 1
    assert res["max_context_ok"] == 4096  # FakeBackend.OOM_TOKENS=5000
    assert res["max_batch_ok"] == 16
    assert any(r.get("oom") for r in res["context"])


def test_cli_fake_end_to_end(tmp_path):
    cli.main(["run", "--backend", "fake", "--quick", "--out", str(tmp_path)])
    d = next(tmp_path.iterdir())
    data = json.loads((d / "results.json").read_text())
    assert data["models"] and (d / "report.md").exists()
    cli.main(["compare", str(d), str(d), "--out", str(tmp_path / "cmp")])
    assert (tmp_path / "cmp" / "comparison.md").exists()
