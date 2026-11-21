# Analiza wydajności inferencji dużych modeli językowych w kontekście dostępnych zasobów GPU

## 1. Cel i zakres
Cel: ustalić, jak wydajność inferencji LLM zależy od GPU (ilość VRAM, przepustowość pamięci, moc obliczeniowa)
oraz od silnika inferencji, i jaki **kontekst** oraz **batch** da się obsłużyć na danym sprzęcie.


## 2. Podstawy inferencji LLM
Generacja autoregresyjna ma dwie fazy o różnym charakterze:

| Faza | Co robi | Wąskie gardło | Metryka |
|---|---|---|---|
| **Prefill** | przetwarza cały prompt równolegle, buduje KV-cache | moc obliczeniowa (FLOPS) | TTFT, prefill tok/s |
| **Decode** | generuje po 1 tokenie, czyta wszystkie wagi + KV-cache | przepustowość pamięci | TPOT, decode tok/s |

**KV-cache** przechowuje klucze i wartości uwagi dla już przetworzonych tokenów, żeby nie liczyć ich od nowa.
Rośnie liniowo z kontekstem i rozmiarem batcha:

`KV na token = 2 · warstwy · kv_heads · head_dim · bajty_na_element`

Przykład (Qwen2.5-7B, fp16: 28 warstw, 4 głowy KV, head_dim 128): 56 KB/token, czyli ok. 1,75 GB dla kontekstu 32k
(batch=1). To właśnie KV-cache, obok wag, ogranicza kontekst i batch na GPU.

**Wagi:** pamięć ≈ liczba parametrów × bajty (fp16/bf16 = 2 B, int8 = 1 B, int4 = 0,5 B). Model 7B w fp16 to ok. 15 GB.

**Szacunek górnej granicy decode (roofline):** w decode dla batch=1 trzeba przeczytać wszystkie wagi na każdy token, więc
`tok/s ≲ przepustowość_pamięci / rozmiar_wag`. Dla 7B fp16 (~15 GB) na GPU o ~1,5 TB/s daje to rzędu 100 tok/s.
Większy batch amortyzuje odczyt wag, więc przepustowość łączna rośnie, aż do ograniczenia obliczeniowego lub pamięciowego (KV).

## 3. Silniki inferencji (przegląd)
| Silnik | Kluczowe cechy | Rola w projekcie |
|---|---|---|
| **HF Transformers** (`generate`) | prosty, uniwersalny, statyczny batching, SDPA/FlashAttention | baseline, zaimplementowany |
| **vLLM** | PagedAttention (stronicowany KV-cache, mniej fragmentacji), continuous batching, prefix caching | główny silnik porównawczy, zaimplementowany |
| **llama.cpp** | formaty GGUF, kwantyzacja int4-int8, CPU i GPU, offload warstw | opcjonalnie (karty 8 GB, kwantyzacja) |
| **SGLang / TGI / TensorRT-LLM** | zoptymalizowane serwery produkcyjne (TRT-LLM tylko NVIDIA) | do przeglądu literaturowego, ewentualnie rozszerzenie |

Do uzupełnienia w trakcie projektu: źródła (artykuł o PagedAttention, FlashAttention, dokumentacje silników).

## 4. Metodologia pomiarów
Implementacja: [`llmbench/`](../llmbench). Wspólny interfejs `Backend` pozwala uruchomić identyczne scenariusze na różnych silnikach.

**Scenariusze**
1. **Latency (batch=1):** prompt 128 tok., generacja 128 tok. Metryki: TTFT, TPOT, decode tok/s. Warm-up 1, 3 powtórzenia, mediana.
2. **Skalowanie batcha:** batch 1-64, prompt 128, generacja 64. Metryki: łączna przepustowość tok/s, TPOT, szczytowa pamięć; stop przy OOM.
3. **Skalowanie kontekstu (batch=1):** 512 do 32k tokenów (do limitu modelu), generacja 32 tok. Metryki: TTFT, prefill tok/s, decode tok/s,
   szczytowa pamięć; **maksymalny kontekst = największy bez OOM**.

**Zasady rzetelności**
- Prompty to losowe tokeny o dokładnej długości, generacja greedy z wymuszoną długością (bez wczesnego EOS), stałe ziarno.
- Prefill mierzony osobno (generacja 1 tokenu); `TPOT = (T_całość − TTFT)/(n−1)`.
- Pamięć: HF mierzony `torch.cuda.max_memory_allocated`; vLLM prealokuje pamięć (`gpu_memory_utilization`), więc dla niego mierzymy pamięć
  urządzenia (NVML), a maksymalny kontekst wynika z limitu `max_model_len` i KV-cache. Wyników pamięci HF i vLLM nie porównujemy wprost.
- Ten sam dtype, wersje bibliotek i sterownik zapisywane w `results.json`; GPU nie powinno być współdzielone w trakcie pomiaru.
- Modele: Qwen2.5-Instruct 0.5B-32B (otwarte, bez logowania), dobierane automatycznie do VRAM (wagi ≤ 70%).

## 5. Zasoby sprzętowe
| Zasób | Szczegóły |
|---|---|
| PLGrid **Athena** (A100) i **Lem** (H100) | granty obliczeniowe |
| Klaster **obelix** (Instytut Informatyki) | dla osób z Instytutu |
| Sprzęt z kartą **24 GB** | dostęp przez dyplomanta prowadzącego (M. Ostaszewski) |
| Sprzęt z kartą **8 GB** | dostęp po mailu |
| Maszyna własna | RTX 5060 Ti 16 GB |

Macierz konfiguracji do pokrycia: ≥ 4 różne GPU (8 / 16 / 24 GB / A100 / H100) × backendy (HF, vLLM) × modele mieszczące się w VRAM.

## 6. Harmonogram (propozycja)
| Etap | Zawartość | Wynik |
|---|---|---|
| 1 | Konta, dostępy, przegląd literatury o silnikach | dostęp do GPU, notatki |
| 2 | Narzędzie `llmbench` (gotowe w wersji początkowej), pierwsze pomiary na maszynie własnej | działający pipeline |
| 3 | Pomiary na 8 GB, 24 GB, Athena, Lem; HF i vLLM | komplet `results/*` |
| 4 | Rozszerzenia: kwantyzacja, KV-cache FP8, multi-GPU (opcjonalnie) | dodatkowe wyniki |
| 5 | Analiza, wnioski, sprawozdanie | raport końcowy |


## 7. Plan analizy wyników i hipotezy
Dane wejściowe: `python -m llmbench compare results/*` (tabele i wykresy).

Hipotezy do sprawdzenia:
1. Decode tok/s (batch=1) skaluje się z przepustowością pamięci GPU i jest zbliżone do granicy roofline.
2. TTFT rośnie superliniowo z kontekstem (koszt uwagi), a prefill tok/s zależy od mocy obliczeniowej.
3. Maksymalny kontekst na danym GPU wyznaczają VRAM − wagi − narzut, podzielone przez KV na token; zgodność z teoretycznym szacunkiem.
4. Przepustowość łączna rośnie z batchem, aż do nasycenia lub OOM; vLLM (continuous batching, PagedAttention) wypada lepiej niż HF przy dużych batchach.
5. Kontekst i batch konkurują o tę samą pamięć KV (kompromis do pokazania na wykresie).

Struktura sprawozdania: wstęp i przegląd silników, metodologia, wyniki per GPU, porównanie GPU, porównanie silników, wnioski (jaki GPU do jakiego modelu i kontekstu), ograniczenia.
