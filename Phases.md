# Hackathon Implementation Roadmap & Execution Phases
## PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine

> **Document:** `Phases.md`  
> **Structure:** 6 Step-by-Step Milestones with Concrete Deliverables, Time Targets & Verification Commands.

---

## Roadmap Overview

```
[ Phase 1: C++ AVX2 SIMD Engine & Manual Indexer ] ──► (Target: 45 mins)
                      │
                      ▼
[ Phase 2: Streaming Audio Ingestion & Ring Buffer ] ──► (Target: 40 mins)
                      │
                      ▼
[ Phase 3: Early Intent & Speculative Trigger ] ──────► (Target: 45 mins)
                      │
                      ▼
[ Phase 4: Deterministic Numeric Guardrail ] ─────────► (Target: 30 mins)
                      │
                      ▼
[ Phase 5: Speculative Neural Vocoder / TTS ] ────────► (Target: 40 mins)
                      │
                      ▼
[ Phase 6: Benchmark Evaluation Suite & Demo ] ───────► (Target: 30 mins)
```

---

## Phase 1: C++ AVX2 SIMD Vector Engine & Knowledge Base Indexer
* **Estimated Time:** 45 minutes
* **Objective:** Parse operational manuals, embed chunks, and compile an ultra-fast C++ AVX2 SIMD vector search DLL.

### Tasks:
1. Parse `manuals/atc_emergency_manual.md` into semantic chunks (Altitude/Separation, Transponder Squawk codes, Reactor MAWP/Venting).
2. Generate 384-dimensional dense embeddings using `all-MiniLM-L6-v2` via [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers).
3. Implement `simd_engine.cpp`:
   - Cache-aligned in-memory vector storage (`_aligned_malloc`).
   - Unrolled AVX2 FMA dot-product kernel (`_mm256_fmadd_ps`) inspired by [ashvardanian/simsimd](https://github.com/ashvardanian/simsimd).
   - SPSC lock-free audio ring buffer struct inspired by [cameron314/readerwriterqueue](https://github.com/cameron314/readerwriterqueue).
4. Compile native dynamic library with MSYS2 `g++`:
   ```bash
   g++ -O3 -mavx2 -mfma -shared -o simd_engine.dll simd_engine.cpp
   ```
5. Build Python ctypes wrapper (`simd_wrapper.py`) to call the C++ DLL.

### Deliverables:
* `manual_indexer.py`
* `simd_engine.cpp`
* `simd_engine.dll`
* `simd_wrapper.py`

### Verification Test:
* Run unit test verifying vector retrieval latency is **< 0.05ms** and returns correct manual chunk for squawk and reactor queries.

---

## Phase 2: Streaming Ingestion & Lock-Free Ring Buffer (Go & C++)
* **Estimated Time:** 40 minutes
* **Objective:** Ingest live 16kHz PCM audio chunks over WebSockets and hand them off to the lock-free circular ring buffer.

### Tasks:
1. Implement `streaming_ws.go`:
   - WebSocket listener using [coder/websocket](https://github.com/coder/websocket) accepting binary 16kHz 16-bit PCM frames (20ms / 640 bytes).
   - Low-overhead memory handoff using `sync.Pool`.
2. Connect Go network stream to the C++ SPSC ring buffer without thread blocking.
3. Integrate Silero VAD ([snakers4/silero-vad](https://github.com/snakers4/silero-vad)) via ONNX Runtime in Python to continuously track speech boundaries with an aggressive 80ms silence endpointing timeout.

### Deliverables:
* `streaming_ws.go`
* `audio_stream_handler.py`
* `vad_detector.py`

### Verification Test:
* Stream `query_01_atc_squawk.wav` in 20ms chunks into the WebSocket server and verify zero frame drops and accurate VAD endpointing.

---

## Phase 3: Early Prefix Intent Detector & Speculative Trigger
* **Estimated Time:** 45 minutes
* **Objective:** Detect operational keywords at $t = 400\text{–}600\text{ms}$ into speech and fire speculative vector retrieval before the utterance ends.

### Tasks:
1. Implement `speculative_engine.py`:
   - Sliding acoustic prefix window operating on partial audio frames.
   - Early keyword spotter:
     - `"squawk"`, `"transponder"`, `"emergency"` $\to$ `transponder_squawk_code`
     - `"reactor"`, `"pressure"`, `"MAWP"`, `"venting"` $\to$ `reactor_mawp_threshold`
   - Confidence thresholding: Dispatches speculative search when $P(\text{intent}) \ge 0.75$.
2. Wire the speculative trigger to query the C++ SIMD engine while audio frames continue streaming.

### Deliverables:
* `early_intent_detector.py`
* `speculative_engine.py`

### Verification Test:
* Test against `query_01` and `query_02` to confirm speculative retrieval fires at $\le 550\text{ms}$ into the 1.5-second audio files.

---

## Phase 4: Deterministic Post-Retrieval Guardrail (Anti-Hallucination)
* **Estimated Time:** 30 minutes
* **Objective:** Guarantee 100% deterministic accuracy for critical numeric data (`7700`, `450.0`, `485.5`) with zero hallucinations.

### Tasks:
1. Implement `numeric_guardrail.py`:
   - Regex/AST parser extracting numbers, units (PSI, feet, nautical miles), and squawk codes from draft answers.
   - Whitelist verification against the retrieved manual chunk.
   - Deterministic replacement: Forces exact values from `ground_truth_numeric_validation.json`:
     - Query 1: Forces General Emergency code **`7700`** (blocks 7500 / 7600).
     - Query 2: Forces MAWP **`450.0 PSI`** and Venting threshold **`485.5 PSI`**.
2. Measure guardrail execution time (must complete in $< 1\text{ms}$).

### Deliverables:
* `numeric_guardrail.py`

### Verification Test:
* Inject intentional hallucinations (e.g. squawk "7777" or pressure "500 PSI") and verify the guardrail intercepts and corrects them to 7700 and 450.0 PSI.

---

## 5. Phase 5: Speculative Neural Vocoder / TTS & Response Caching
* **Estimated Time:** 40 minutes
* **Objective:** Pre-synthesize the verified response into raw 16kHz PCM audio while the user is finishing speaking and release it on silence.

### Tasks:
1. Integrate Piper TTS via [rhasspy/piper](https://github.com/rhasspy/piper) (VITS ONNX architecture) in `tts_streamer.py`.
2. Implement speculative response caching (adapting [NVIDIA/voice-agent-examples](https://github.com/NVIDIA/voice-agent-examples)):
   - As soon as the draft response is verified, start synthesizing the first audio frame into a memory buffer.
   - When VAD triggers Speech-Off ($80\text{ms}$ silence), immediately flush the pre-rendered audio packets to the client.
3. Validate total post-speech turnaround latency ($T_{\text{post-speech}} < 250\text{ms}$).

### Deliverables:
* `tts_streamer.py`
* `response_cacher.py`

### Verification Test:
* Measure timestamp from user silence detection to first audio byte emitted; confirm $< 120\text{ms}$.

---

## Phase 6: Automated Benchmark Suite & Interactive Demo UI
* **Estimated Time:** 30 minutes
* **Objective:** Provide a production-grade evaluation runner and visual demonstration for the judges.

### Tasks:
1. Implement `evaluate_benchmark.py`:
   - Ingests `ground_truth_numeric_validation.json`.
   - Simulates live microphone streaming for `query_01_atc_squawk.wav` and `query_02_reactor_pressure.wav`.
   - Records microsecond-level timing for every stage:
     $T_{\text{buf}}$, $T_{\text{intent}}$, $T_{\text{simd}}$, $T_{\text{numeric\_val}}$, and $T_{\text{TTFA}}$.
   - Asserts sub-250ms latency and 100% numeric match; prints an ASCII compliance scorecard with **PASS** badges.
2. Build an interactive demonstration dashboard (`demo_app.py` / Web UI) showing the live audio wave, real-time speculative trigger timeline, and instant audio playback.

### Deliverables:
* `evaluate_benchmark.py`
* `demo_app.py`
* `BENCHMARK_RESULTS.md`

### Verification Test:
* Run `python evaluate_benchmark.py` and verify all tests pass with green checkmarks!
