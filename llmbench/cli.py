"""CLI:  python -m llmbench {info,run,report,compare}"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import yaml

from . import bench, gpu
from .backends import make_backend, resolve_dtype
from .report import load_results, write_comparison, write_report

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "configs" / "default.yaml").read_text())


def cmd_info(a):
    print(json.dumps({"gpu": gpu.detect_gpu(a.device).to_dict(), "system": gpu.system_info()}, indent=2))


def cmd_run(a):
    cfg = load_config(a.config)
    backend = a.backend or cfg["backend"]
    info = gpu.detect_gpu(a.device)
    if backend != "fake" and not info.memory_total_gb:
        sys.exit("Nie wykryto GPU. Użyj --backend fake do próbnego uruchomienia bez GPU.")
    if backend == "fake" and not info.memory_total_gb:
        info.memory_total_gb = 24.0
    if a.quick:  # krótki przebieg do sprawdzenia, że wszystko działa
        cfg["latency"].update(new_tokens=32, repeats=1)
        cfg["batch"].update(sizes=[1, 4, 16], new_tokens=16)
        cfg["context"].update(lengths=[512, 2048, 8192], new_tokens=8)
    dtype = a.dtype or cfg["dtype"]
    bpp = {"fp32": 4.0}.get(dtype, 2.0)
    models = cfg["models"]
    if a.models:
        known = {m["id"]: m for m in models}
        models = [known.get(x, {"id": x, "params_b": 0.0}) for x in a.models]
        skipped = []
    else:
        chosen = bench.select_models(models, info.memory_total_gb, cfg["max_weights_fraction"], bpp)
        skipped = [m["id"] for m in models if m not in chosen]
        models = chosen
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(a.out or ROOT / "results") / f"{info.slug}_{backend}_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"GPU: {info.name} ({info.memory_total_gb:.1f} GB, cc {info.compute_capability})")
    print(f"Backend: {backend} | modele: {[m['id'] for m in models]}\nWyniki: {outdir}")

    be = make_backend(backend, dtype=dtype, device_index=a.device, **(cfg.get("vllm") or {}))
    rng = random.Random(cfg["seed"])
    result = {
        "gpu": info.to_dict(), "system": gpu.system_info(),
        "meta": {"backend": backend, "dtype": resolve_dtype(dtype, info.supports_bf16),
                 "timestamp": datetime.now().isoformat(timespec="seconds"), "config": cfg},
        "models": [], "skipped_models": skipped,
    }
    for m in models:
        print(f"[{m['id']}]")
        result["models"].append(bench.run_model(be, m, cfg, rng))
        (outdir / "results.json").write_text(json.dumps(result, indent=2))  # zapis po każdym modelu
    print("Raport:", write_report(result, outdir))


def cmd_report(a):
    for r in load_results([a.path]):
        print(write_report(r, Path(a.path)))


def cmd_compare(a):
    res = load_results(a.paths)
    if not res:
        sys.exit("Brak wyników (results.json).")
    print(write_comparison(res, Path(a.out)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="llmbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="wykryj GPU i środowisko"); s.add_argument("--device", type=int, default=0)
    s.set_defaults(f=cmd_info)
    s = sub.add_parser("run", help="uruchom benchmark i wygeneruj raport")
    s.add_argument("--config"); s.add_argument("--backend", choices=["hf", "vllm", "fake"])
    s.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"])
    s.add_argument("--models", nargs="+", help="nadpisz listę modeli (HF id)")
    s.add_argument("--device", type=int, default=0); s.add_argument("--out")
    s.add_argument("--quick", action="store_true", help="skrócony przebieg (smoke test)")
    s.set_defaults(f=cmd_run)
    s = sub.add_parser("report", help="odtwórz raport z results.json"); s.add_argument("path")
    s.set_defaults(f=cmd_report)
    s = sub.add_parser("compare", help="porównaj wyniki z wielu GPU")
    s.add_argument("paths", nargs="+"); s.add_argument("--out", default=str(ROOT / "results" / "comparison"))
    s.set_defaults(f=cmd_compare)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
