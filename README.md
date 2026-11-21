# Analiza wydajności inferencji LLM vs. zasoby GPU

Benchmark mierzący: **TTFT** (prefill), **TPOT / tok/s** (decode), **skalowanie przepustowości z batchem**,
**maksymalny użyteczny kontekst** (do OOM) i zużycie pamięci - dla różnych GPU i silników (HF Transformers, vLLM).
Narzędzie samo wykrywa GPU, dobiera modele do VRAM-u i zapisuje raport.

## Szybki start
```bash
./scripts_setup.sh            # venv + zależności (./scripts_setup.sh vllm dodaje vLLM)
source .venv/bin/activate
python -m llmbench info                       # wykryte GPU / środowisko
python -m llmbench run --backend fake --quick # test bez GPU (symulator)
python -m llmbench run                        # pełny benchmark na wykrytym GPU (backend hf)
python -m llmbench run --backend vllm         # ten sam benchmark na vLLM
python -m llmbench run --models Qwen/Qwen2.5-7B-Instruct --quick
python -m llmbench compare results/*_hf_*     # porównanie wielu GPU -> results/comparison/comparison.md
pytest                                        # testy logiki
```
Wynik: `results/<gpu-slug>_<backend>_<czas>/{results.json,report.md,*.png}`.
Wybór GPU na maszynie wielokartowej: `--device N` (lub `CUDA_VISIBLE_DEVICES`).

## Jak to działa
- `llmbench/gpu.py` - detekcja (NVML → nvidia-smi → torch): nazwa, VRAM, compute capability, sterownik, bf16.
- `llmbench/backends.py` - wspólny interfejs `Backend` (HF, vLLM, fake). Dodanie np. llama.cpp = 1 klasa.
- `llmbench/bench.py` - scenariusze: latency (b=1), batch scaling, context scaling; OOM jest łapany i raportowany
  jako granica (max batch / max kontekst). Prompty to losowe tokeny o dokładnej długości, generacja greedy bez EOS.
- Prefill mierzony osobno (generacja 1 tokenu), `TPOT = (T_całość − TTFT)/(n−1)`.
- Modele dobierane automatycznie: wagi ≤ 70% VRAM (`max_weights_fraction`). Nadpisanie: `--models`.
- Pamięć: HF → `torch.cuda.max_memory_allocated`; vLLM prealokuje (`gpu_memory_utilization`), więc
  mierzona jest pamięć urządzenia (NVML), a max kontekst wynika z limitu `max_model_len`/KV-cache.
- Raport zawiera też teoretyczny rozmiar KV-cache na token (`2·L·kv_heads·head_dim·bajty`).

## Dostępne zasoby (z maila prowadzącego)
| Zasób | Jak |
|---|---|
| PLGrid **Athena** (A100) i **Lem** (H100) | założyć konto PLGrid, podać prowadzącego jako opiekuna; `sbatch slurm/run_athena.sh` (uzupełnić grant `-A`) |
| Klaster Instytutu Informatyki **obelix** | promotor wnioskuje o dostęp |
| Sprzęt z kartą **24 GB** | kontakt: Mateusz Ostaszewski (dyplomant prowadzącego), ustalić termin |
| Sprzęt z kartą **8 GB** | mail od 18.12 → dostęp |

Na każdej maszynie: `git clone`, `./scripts_setup.sh`, `python -m llmbench run`, potem zebrać katalogi
`results/*` w jedno miejsce i `python -m llmbench compare results/*`.
Uwaga: na 8 GB automatycznie zostaną modele ≤1.5B; modele gated (Llama) wymagają `huggingface-cli login`.

## Pomysły na rozszerzenie analizy
vLLM vs HF (continuous batching, PagedAttention), kwantyzacja (AWQ/GPTQ/FP8), KV-cache FP8,
multi-GPU (tensor parallel) na Athenie/Lem, wpływ limitu mocy GPU.
