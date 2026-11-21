"""Scenariusze pomiarowe: latency (TTFT/TPOT), skalowanie batcha, skalowanie kontekstu."""
from __future__ import annotations

import random
import statistics as st
import time

from .backends import Backend, OOMError


def make_prompts(vocab: int, batch: int, length: int, rng: random.Random) -> list[list[int]]:
    hi = max(100, min(vocab, 30000))
    return [[rng.randrange(100, hi) for _ in range(length)] for _ in range(batch)]


def _timed(be: Backend, prompts, n: int) -> float:
    t = time.perf_counter()
    be.generate(prompts, n)
    return time.perf_counter() - t


def measure(be: Backend, prompts, new_tokens: int, repeats: int = 1, warmup: int = 0) -> dict:
    """Prefill mierzony osobno (1 token), decode = (całość - prefill)/(n-1)."""
    for _ in range(warmup):
        be.generate(prompts, 2)
    be.reset_peak()
    pre, tot = [], []
    for _ in range(repeats):
        pre.append(_timed(be, prompts, 1))
        tot.append(_timed(be, prompts, new_tokens))
    b, n = len(prompts), len(prompts[0])
    ttft, total = st.median(pre), st.median(tot)
    decode = max(total - ttft, 1e-9)
    steps = max(new_tokens - 1, 1)
    return {
        "batch": b, "prompt_len": n, "new_tokens": new_tokens,
        "ttft_s": ttft,
        "tpot_ms": decode / steps * 1000,
        "decode_tok_s": b * steps / decode,             # przepustowość generacji
        "prefill_tok_s": b * n / ttft,
        "e2e_s": total,
        "throughput_tok_s": b * new_tokens / total,
        "stdev_total_s": st.pstdev(tot) if len(tot) > 1 else 0.0,
        "peak_mem_gb": be.peak_mem_gb(),
    }


def run_latency(be, cfg, rng):
    c = cfg["latency"]
    p = make_prompts(be.vocab_size, 1, c["prompt_len"], rng)
    return measure(be, p, c["new_tokens"], c["repeats"], c["warmup"])


def run_batch_scaling(be, cfg, rng):
    c, rows = cfg["batch"], []
    for b in c["sizes"]:
        try:
            rows.append(measure(be, make_prompts(be.vocab_size, b, c["prompt_len"], rng),
                                c["new_tokens"], repeats=2, warmup=1 if not rows else 0))
        except OOMError:
            rows.append({"batch": b, "oom": True})
            break
    return rows


def run_context_scaling(be, cfg, rng, limit: int | None):
    c, rows = cfg["context"], []
    cap = min(x for x in (limit, c.get("max_model_len_cap")) if x)
    for n in c["lengths"]:
        if n + c["new_tokens"] > cap:
            break
        try:
            rows.append(measure(be, make_prompts(be.vocab_size, 1, n, rng), c["new_tokens"],
                                repeats=1, warmup=0))
        except OOMError:
            rows.append({"batch": 1, "prompt_len": n, "oom": True})
            break
    return rows


def kv_bytes_per_token(model_id: str, dtype_bytes: int = 2) -> int | None:
    """Teoretyczny rozmiar KV-cache na token: 2 * warstwy * kv_heads * head_dim * bajty."""
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(model_id)
        heads = c.num_attention_heads
        kv = getattr(c, "num_key_value_heads", heads)
        hd = getattr(c, "head_dim", None) or c.hidden_size // heads
        return 2 * c.num_hidden_layers * kv * hd * dtype_bytes
    except Exception:
        return None


def select_models(models: list[dict], vram_gb: float, frac: float, bytes_per_param: float = 2.0):
    return [m for m in models if m["params_b"] * bytes_per_param <= frac * vram_gb]


def run_model(be: Backend, model: dict, cfg: dict, rng: random.Random, log=print) -> dict:
    mid = model["id"]
    res: dict = {"model": mid, "params_b": model["params_b"], "status": "ok"}
    cap = cfg["context"]["max_model_len_cap"]
    need = max(cfg["latency"]["prompt_len"] + cfg["latency"]["new_tokens"],
               cfg["batch"]["prompt_len"] + cfg["batch"]["new_tokens"])
    max_len = max(cap, need)
    t = time.perf_counter()
    try:
        be.load(mid, max_len)
    except OOMError as e:
        res.update(status="oom_on_load", error=str(e)[:300])
        log(f"  OOM podczas ładowania {mid}")
        return res
    res["load_time_s"] = time.perf_counter() - t
    res["weights_mem_gb"] = be.weights_mem_gb()
    res["dtype"] = be.dtype
    res["max_position"] = be.max_position
    res["kv_bytes_per_token"] = kv_bytes_per_token(mid, 2 if be.dtype != "fp32" else 4)
    try:
        for key, fn in (("latency", lambda: run_latency(be, cfg, rng)),
                        ("batch", lambda: run_batch_scaling(be, cfg, rng)),
                        ("context", lambda: run_context_scaling(be, cfg, rng, be.max_position))):
            log(f"  {key} ...")
            try:
                res[key] = fn()
            except OOMError as e:
                res[key] = None
                res.setdefault("errors", {})[key] = str(e)[:300]
        ctx_ok = [r["prompt_len"] for r in res.get("context") or [] if not r.get("oom")]
        res["max_context_ok"] = max(ctx_ok) if ctx_ok else 0
        res["max_batch_ok"] = max([r["batch"] for r in res.get("batch") or [] if not r.get("oom")],
                                  default=0)
    finally:
        be.unload()
    return res
