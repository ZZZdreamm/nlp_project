"""Backendy inferencji. Wspólny interfejs pozwala porównywać silniki (HF vs vLLM)."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

from . import gpu as gpumod


class OOMError(Exception):
    """Brak pamięci GPU (lub kontekst ponad możliwości silnika)."""


class Backend(ABC):
    name = "base"

    def __init__(self, dtype: str = "auto", device_index: int = 0, **opts):
        self.dtype, self.device_index, self.opts = dtype, device_index, opts

    @abstractmethod
    def load(self, model_id: str, max_model_len: int) -> None: ...

    @abstractmethod
    def generate(self, prompts: list[list[int]], new_tokens: int) -> None:
        """Generuje dokładnie `new_tokens` tokenów (greedy, bez EOS) dla każdego promptu."""

    @abstractmethod
    def unload(self) -> None: ...

    def reset_peak(self) -> None: ...

    def peak_mem_gb(self) -> float | None:
        return None

    def weights_mem_gb(self) -> float | None:
        return None

    @property
    def vocab_size(self) -> int:
        return 32000

    @property
    def max_position(self) -> int | None:
        return None


def resolve_dtype(dtype: str, supports_bf16: bool) -> str:
    return ("bf16" if supports_bf16 else "fp16") if dtype == "auto" else dtype


class HFBackend(Backend):
    name = "hf"

    def load(self, model_id, max_model_len):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        self.torch = torch
        info = gpumod.detect_gpu(self.device_index)
        self.dtype = resolve_dtype(self.dtype, info.supports_bf16)
        td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]
        self.device = f"cuda:{self.device_index}"
        self._cfg = AutoConfig.from_pretrained(model_id)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=td, attn_implementation="sdpa").to(self.device).eval()
        except torch.cuda.OutOfMemoryError as e:
            self.unload()
            raise OOMError(str(e)) from e
        torch.cuda.synchronize()
        self._weights = torch.cuda.memory_allocated(self.device) / 1024**3

    def generate(self, prompts, new_tokens):
        torch = self.torch
        ids = torch.tensor(prompts, device=self.device)
        try:
            with torch.inference_mode():
                self.model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=new_tokens, min_new_tokens=new_tokens, pad_token_id=0)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            raise OOMError(str(e)) from e

    def unload(self):
        import gc

        self.__dict__.pop("model", None)
        gc.collect()
        if hasattr(self, "torch"):
            self.torch.cuda.empty_cache()

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_mem_gb(self):
        return self.torch.cuda.max_memory_allocated(self.device) / 1024**3

    def weights_mem_gb(self):
        return self._weights

    @property
    def vocab_size(self):
        return self._cfg.vocab_size

    @property
    def max_position(self):
        return getattr(self._cfg, "max_position_embeddings", None)


class VLLMBackend(Backend):
    name = "vllm"

    def load(self, model_id, max_model_len):
        from transformers import AutoConfig
        from vllm import LLM

        info = gpumod.detect_gpu(self.device_index)
        self.dtype = resolve_dtype(self.dtype, info.supports_bf16)
        self._cfg = AutoConfig.from_pretrained(model_id)
        before = gpumod.used_memory_gb(self.device_index) or 0
        try:
            self.llm = LLM(
                model_id, dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[self.dtype],
                max_model_len=max_model_len,
                gpu_memory_utilization=self.opts.get("gpu_memory_utilization", 0.9),
                enforce_eager=self.opts.get("enforce_eager", False),
                seed=0)
        except (ValueError, RuntimeError, MemoryError) as e:
            raise OOMError(str(e)) from e
        self._used = (gpumod.used_memory_gb(self.device_index) or 0) - before

    def generate(self, prompts, new_tokens):
        from vllm import SamplingParams

        sp = SamplingParams(temperature=0, max_tokens=new_tokens, min_tokens=new_tokens,
                            ignore_eos=True)
        try:
            self.llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        except (ValueError, RuntimeError) as e:
            raise OOMError(str(e)) from e

    def unload(self):
        import gc

        self.__dict__.pop("llm", None)
        gc.collect()
        try:
            import torch
            from vllm.distributed.parallel_state import destroy_model_parallel

            destroy_model_parallel()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def peak_mem_gb(self):  # vLLM prealokuje pamięć -> zajęta pamięć urządzenia
        return gpumod.used_memory_gb(self.device_index)

    def weights_mem_gb(self):
        return self._used

    @property
    def vocab_size(self):
        return self._cfg.vocab_size

    @property
    def max_position(self):
        return getattr(self._cfg, "max_position_embeddings", None)


class FakeBackend(Backend):
    """Symulator do testów i 'dry-run' bez GPU. Prefill ~ liniowy, decode ~ stały, OOM > limit."""
    name = "fake"
    OOM_TOKENS = 5000

    def load(self, model_id, max_model_len):
        self.max_model_len = max_model_len

    def generate(self, prompts, new_tokens):
        b, n = len(prompts), len(prompts[0])
        if b * (n + new_tokens) > self.OOM_TOKENS:
            raise OOMError("fake OOM")
        time.sleep(1e-6 * b * n + 1e-4 * new_tokens)

    def unload(self): ...

    def peak_mem_gb(self):
        return 1.0

    def weights_mem_gb(self):
        return 1.0

    @property
    def max_position(self):
        return 32768


def make_backend(name: str, **kw) -> Backend:
    return {"hf": HFBackend, "vllm": VLLMBackend, "fake": FakeBackend}[name](**kw)
