"""Generowanie raportów: Markdown + wykresy PNG dla jednego GPU oraz porównanie wielu GPU."""
from __future__ import annotations

import json
from pathlib import Path


def _f(x, spec=".1f", dash="-"):
    return dash if x is None else format(x, spec)


def _plot_model_set(models, outdir: Path, title: str, tag: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []

    def fig(kind, xkey, ylabel, ykey, fname, xlabel, logx=False):
        plt.figure(figsize=(6, 4))
        plotted = False
        for m in models:
            rows = [r for r in (m.get(kind) or []) if not r.get("oom") and r.get(ykey) is not None]
            if rows:
                plt.plot([r[xkey] for r in rows], [r[ykey] for r in rows], marker="o",
                         label=m["model"].split("/")[-1])
                plotted = True
        if plotted:
            if logx:
                plt.xscale("log", base=2)
            plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.3)
            plt.legend(fontsize=7); plt.tight_layout()
            plt.savefig(outdir / fname, dpi=130)
            files.append(fname)
        plt.close()

    fig("batch", "batch", "tok/s (łącznie)", "throughput_tok_s", f"{tag}batch_throughput.png",
        "rozmiar batcha", True)
    fig("context", "prompt_len", "TTFT [s]", "ttft_s", f"{tag}context_ttft.png", "długość kontekstu", True)
    fig("context", "prompt_len", "decode tok/s", "decode_tok_s", f"{tag}context_decode.png",
        "długość kontekstu", True)
    fig("context", "prompt_len", "szczytowa pamięć [GB]", "peak_mem_gb", f"{tag}context_mem.png",
        "długość kontekstu", True)
    return files


def write_report(result: dict, outdir: Path) -> Path:
    outdir = Path(outdir)
    g, s, meta = result["gpu"], result["system"], result["meta"]
    L = [f"# Raport wydajności inferencji LLM - {g['name']}", ""]
    L += ["## Środowisko", "",
          f"- **GPU:** {g['name']} x{g.get('count', 1)} (używane: #{g['index']}), "
          f"{g['memory_total_gb']:.1f} GB VRAM, compute capability {g['compute_capability']}",
          f"- **Sterownik / CUDA:** {g['driver']} / {g['cuda_runtime'] or '-'}",
          f"- **Backend:** {meta['backend']} ({meta['dtype']}), data: {meta['timestamp']}",
          f"- **Host:** {s['hostname']} | {s['cpu_cores']} rdzeni CPU | {s['ram_gb']} GB RAM",
          f"- **Pakiety:** " + ", ".join(f"{k} {v}" for k, v in s["packages"].items() if v), ""]
    models = result["models"]
    L += ["## Podsumowanie", "",
          "| Model | Status | Wagi [GB] | KV [KB/tok] | Decode tok/s (b=1) | TTFT [s] | TPOT [ms] "
          "| Max batch | Max kontekst |", "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        lat = m.get("latency") or {}
        kv = m.get("kv_bytes_per_token")
        L.append(f"| {m['model'].split('/')[-1]} | {m['status']} | {_f(m.get('weights_mem_gb'))} | "
                 f"{_f(kv and kv / 1024)} | {_f(lat.get('decode_tok_s'))} | {_f(lat.get('ttft_s'), '.3f')} | "
                 f"{_f(lat.get('tpot_ms'))} | {m.get('max_batch_ok', '-')} | {m.get('max_context_ok', '-')} |")
    if result.get("skipped_models"):
        L += ["", "Pominięte (wagi nie mieszczą się w VRAM): " + ", ".join(result["skipped_models"])]
    for m in models:
        L += ["", f"## {m['model']}", ""]
        if m["status"] != "ok":
            L += [f"Status: **{m['status']}** {m.get('error', '')}"]
            continue
        L += ["### Skalowanie batcha", "", "| batch | tok/s | TPOT [ms] | TTFT [s] | peak mem [GB] |", "|---|---|---|---|---|"]
        for r in m.get("batch") or []:
            L.append(f"| {r['batch']} | OOM | | | |" if r.get("oom") else
                     f"| {r['batch']} | {r['throughput_tok_s']:.1f} | {r['tpot_ms']:.1f} | "
                     f"{r['ttft_s']:.3f} | {_f(r['peak_mem_gb'], '.2f')} |")
        L += ["", "### Skalowanie kontekstu (batch=1)", "",
              "| kontekst | TTFT [s] | prefill tok/s | decode tok/s | peak mem [GB] |", "|---|---|---|---|---|"]
        for r in m.get("context") or []:
            L.append(f"| {r['prompt_len']} | OOM | | | |" if r.get("oom") else
                     f"| {r['prompt_len']} | {r['ttft_s']:.3f} | {r['prefill_tok_s']:.0f} | "
                     f"{r['decode_tok_s']:.1f} | {_f(r['peak_mem_gb'], '.2f')} |")
        if m.get("errors"):
            L += ["", "Błędy: " + "; ".join(f"{k}: {v[:80]}" for k, v in m["errors"].items())]
    plots = _plot_model_set(models, outdir, g["name"], "")
    if plots:
        L += ["", "## Wykresy", ""] + [f"![{p}]({p})" for p in plots]
    path = outdir / "report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def load_results(paths) -> list[dict]:
    out = []
    for p in map(Path, paths):
        f = p / "results.json" if p.is_dir() else p
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def write_comparison(results: list[dict], outdir: Path) -> Path:
    """Porównanie wielu GPU: dla każdego modelu tabela GPU x metryki + wykresy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    by_model: dict[str, list] = {}
    for r in results:
        for m in r["models"]:
            if m["status"] == "ok":
                by_model.setdefault(m["model"], []).append((r["gpu"], r["meta"]["backend"], m))
    L = ["# Porównanie GPU", "", "GPU: " + ", ".join(f"{r['gpu']['name']} ({r['meta']['backend']})" for r in results), ""]
    plots = []
    for model, entries in by_model.items():
        short = model.split("/")[-1]
        L += [f"## {short}", "", "| GPU | backend | VRAM [GB] | decode tok/s (b=1) | TTFT [s] | max tok/s (batch) | max batch | max kontekst |",
              "|---|---|---|---|---|---|---|---|"]
        for g, be, m in entries:
            lat = m.get("latency") or {}
            best = max([r["throughput_tok_s"] for r in m.get("batch") or [] if not r.get("oom")], default=None)
            L.append(f"| {g['name']} | {be} | {g['memory_total_gb']:.0f} | {_f(lat.get('decode_tok_s'))} | "
                     f"{_f(lat.get('ttft_s'), '.3f')} | {_f(best)} | {m.get('max_batch_ok')} | {m.get('max_context_ok')} |")
        L.append("")
        plt.figure(figsize=(6, 4))
        for g, be, m in entries:
            rows = [r for r in m.get("batch") or [] if not r.get("oom")]
            if rows:
                plt.plot([r["batch"] for r in rows], [r["throughput_tok_s"] for r in rows], marker="o",
                         label=f"{g['name']} ({be})")
        plt.xscale("log", base=2); plt.xlabel("batch"); plt.ylabel("tok/s"); plt.title(short)
        plt.grid(alpha=.3); plt.legend(fontsize=7); plt.tight_layout()
        fn = f"cmp_{short}.png"; plt.savefig(outdir / fn, dpi=130); plt.close()
        plots.append(fn)
        L.append(f"![{short}]({fn})\n")
    path = outdir / "comparison.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
