# Master Execution Blueprint & Final Plan
## PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine

> **Document:** `Final Plan.md`  
> **Domain:** Streaming Multimedia / Speech Processing / Low-Latency RAG  
> **Production Target:** Round-trip response latency strictly **< 250 milliseconds** (from user speech completion to first synthesized audio frame) with **100% deterministic numeric accuracy** (zero hallucinations on altitudes, squawk codes, pressures).

---

## 1. Executive Summary & Winning Strategy

Conventional conversational voice systems fail high-stakes operational emergencies because their sequential cascade (*Speech-to-Text → Text Embedding → Vector DB Search → LLM Inference → Text-to-Speech*) introduces **1.8 to 4.0 seconds** of lag.

Our winning strategy decouples the pipeline using **Speculative Execution ("Listen-While-Thinking")**:
1. **Continuous 16kHz PCM Ingestion:** Streamed in 20ms chunks over a concurrent Go WebSocket server into a lock-free C++ ring buffer.
2. **Early Intent Detection (400ms – 600ms):** A rolling prefix acoustic classifier recognizes operational keywords while the user is still speaking.
3. **SIMD-Accelerated Vector Retrieval (< 0.02ms):** A native C++ AVX2 engine searches cache-aligned operational manual embeddings before the utterance concludes.
4. **Deterministic Numeric Guardrail:** Strict AST/Regex validation forces exact figures (`7700`, `450.0`, `485.5`) against manual ground truth, preventing any LLM hallucination.
5. **Speculative TTS Pre-Rendering:** Piper VITS synthesizes response audio into a memory ring buffer while speech continues.
6. **Instant Audio Release (~85ms):** When Silero VAD detects speech end (80ms silence), pre-rendered audio packets are immediately flushed to the wire.

---

## 2. Core Constraints & Language Division

As mandated by the problem statement:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ C++ (SIMD & Data Structures)                                                │
│ • Handcrafted AVX2 vector dot-product kernel (_mm256_fmadd_ps)              │
│ • Contiguous float vector index stored in CPU cache memory                  │
│ • Lock-free SPSC circular ring buffer for 16kHz PCM frames                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ Native bindings / C-API
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Go (Concurrent Networking Layer)                                            │
│ • High-throughput WebSocket server (`coder/websocket` or Gorilla)           │
│ • Full-duplex binary streaming of 20ms PCM audio frames                     │
│ • Minimal allocation, sub-millisecond network handoff                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ Local IPC / In-Process Orchestration
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Python (Acoustics, Intelligence & Benchmark Orchestration)                  │
│ • Silero VAD (Continuous speech tracking, 80ms silence endpointing)         │
│ • Rolling prefix intent classifier (triggers at 400ms–600ms)                │
│ • Speculative response drafter & deterministic numeric guardrail            │
│ • Piper TTS (VITS ONNX) streaming neural vocoder                            │
│ • Automated benchmark test runner (`evaluate_benchmark.py`)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Benchmark Dataset & Ground Truth Alignment

The system must achieve 100% precision and pass all numeric tests on the provided benchmark package:

### 1. Query 1 (`audio_samples/query_01_atc_squawk.wav`)
* **Audio Format:** 16,000 Hz, 16-bit Mono PCM, 24,000 frames (1.50 seconds).
* **Ground Truth Intent:** `transponder_squawk_code`
* **Target Knowledge Source:** `manuals/atc_emergency_manual.md` (Section 1)
* **Expected Numeric Values:** `[7700]`
* **Guardrail Constraint:** Must return squawk code `7700` for General Emergency. Must never output 7500 (Hijacking) or 7600 (Radio Failure) for this query.

### 2. Query 2 (`audio_samples/query_02_reactor_pressure.wav`)
* **Audio Format:** 16,000 Hz, 16-bit Mono PCM, 24,000 frames (1.50 seconds).
* **Ground Truth Intent:** `reactor_mawp_threshold`
* **Target Knowledge Source:** `manuals/atc_emergency_manual.md` (Section 2)
* **Expected Numeric Values:** `[450.0, 485.5]`
* **Guardrail Constraint:** Must return Core R-101 MAWP of `450.0 PSI` and Emergency Venting Trip Threshold of `485.5 PSI`.

---

## 4. Latency Budget Allocation (Target: < 250ms)

$$\text{Post-Speech Latency} = T_{\text{VAD\_hangover}} + T_{\text{commit\_check}} + T_{\text{wire\_flush}}$$

| Stage | Execution Timeline | Internal Latency | Post-Speech Contribution |
| :--- | :--- | :--- | :--- |
| **PCM Ingestion & Ring Buffer** | Continuous ($0.0\text{s} \to 1.5\text{s}$) | $\approx 0.5\text{ms}$ | $0\text{ms}$ (Concurrent) |
| **Early Prefix Intent Trigger** | At $t = 450\text{ms}$ into speech | $\approx 10\text{ms}$ | $0\text{ms}$ (Hidden behind speech) |
| **C++ AVX2 SIMD Vector Search** | At $t = 460\text{ms}$ into speech | **$< 0.02\text{ms}$** | $0\text{ms}$ (Hidden behind speech) |
| **Deterministic Numeric Guardrail** | At $t = 461\text{ms}$ into speech | $\approx 0.3\text{ms}$ | $0\text{ms}$ (Hidden behind speech) |
| **Speculative TTS Pre-Rendering** | At $t = 462\text{ms} \to 500\text{ms}$ | $\approx 35\text{ms}$ | $0\text{ms}$ (Pre-cached in memory) |
| **VAD Speech-Off Detection** | At $t = 1.50\text{s}$ (Speech ends) | **$80\text{ms}$** | **$80\text{ms}$** |
| **Audio Packet Egress to Client** | At $t = 1.58\text{s}$ | $\approx 5\text{ms}$ | **$5\text{ms}$** |
| **TOTAL TURNAROUND TIME** | | | **$\mathbf{\approx 85\text{ ms}}$ (PASS < 250ms)** |

---

## 5. Key Deliverables & Artifacts

1. **`simd_engine.cpp` & `simd_engine.dll`:** C++ AVX2 SIMD vector search library and circular ring buffer.
2. **`speculative_engine.py`:** Streaming acoustic chunk reader, Silero VAD, and rolling prefix intent predictor.
3. **`numeric_guardrail.py`:** Deterministic AST/Regex validator enforcing exact ground truth numbers.
4. **`streaming_ws.go`:** Concurrent Go WebSocket server for bidirectional audio streaming.
5. **`evaluate_benchmark.py`:** Automated test suite that streams `query_01` and `query_02` frame-by-frame, logs microsecond timestamps, and validates ground truth.
6. **`demo_app.py` / Web UI:** Interactive dashboard visualizing speculative pre-retrieval in real time with audio playback.

---

## 6. Success & Evaluation Criteria

| Evaluation Check | Target Metric | Verification Method |
| :--- | :--- | :--- |
| **Response Latency** | **< 250ms** | End-of-speech to First Audio Frame timestamp in `evaluate_benchmark.py` |
| **Query 1 Accuracy** | **100% Match** | Exact presence of numeric `7700` and absence of hallucinations |
| **Query 2 Accuracy** | **100% Match** | Exact presence of numerics `450.0` and `485.5` PSI |
| **Audio Stream Integrity** | **0 glitches / drops** | Continuous 16kHz PCM playback verified via soundfile/sounddevice |
| **C++ SIMD Acceleration** | **Active AVX2** | CPU instruction trace confirms `_mm256_fmadd_ps` execution |
