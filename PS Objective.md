# PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine
## Comprehensive Objectives, Features & Verification Checklist

> **Hackathon Track:** Streaming Multimedia / Speech Processing / Low-Latency RAG  
> **Target Outcome:** High-stakes voice assistant capable of bidirectional streaming audio retrieval with guaranteed sub-250ms latency and 100% deterministic numeric accuracy.

---

## 1. 🏗️ Architectural & Pipeline Objectives

- [ ] **Bidirectional Streaming Audio Pipeline**
  - [ ] Support live full-duplex audio stream over network protocol (WebSocket / WebRTC).
  - [ ] Continuous chunked 16kHz 16-bit PCM ingestion (20ms – 50ms audio frames).
  - [ ] Streaming audio egress: Deliver synthesized speech in chunked packets back to the client.

- [ ] **High-Concurrency Ingestion & Lock-Free Ring Buffer**
  - [ ] Implement low-latency circular ring buffer for incoming audio frames.
  - [ ] Zero-copy / minimal-copy buffer management to avoid memory reallocation during streaming.
  - [ ] Concurrent networking layer handling client connections without blocking the inference loop.

- [ ] **Streaming Acoustic Ingestion & Early Prefix Detection**
  - [ ] Continuous Voice Activity Detection (VAD) with aggressive low-latency endpointing (≤ 80–100ms silence detection).
  - [ ] Streaming phoneme / acoustic feature extraction.
  - [ ] Early keyword / intent predictor operating on partial audio fragments (400ms – 600ms into speech).

---

## 2. ⚡ Speculative Decoding & Retrieval Engine

- [ ] **In-Speech Speculative Triggering**
  - [ ] Trigger retrieval *before* the user has completed their query sentence (predictive execution).
  - [ ] Maintain an early intent classifier with confidence scoring ($\tau \ge 0.75$).
  - [ ] Ability to speculatively query the knowledge base while the audio stream is still actively recording.

- [ ] **Draft Response Pipelining**
  - [ ] Pre-fetch and pre-rank candidate operational manual chunks during user speech.
  - [ ] Pre-generate draft response tokens into an execution buffer.
  - [ ] Seamlessly commit or invalidate speculative drafts once final speech endpointing confirms the utterance.

---

## 3. 🚀 SIMD-Accelerated Vector Search (In-Memory Index)

- [ ] **SIMD Hardware Acceleration (AVX2 / C++)**
  - [ ] Implement vectorized dot-product / cosine similarity in C++ using AVX2 intrinsics (`_mm256_fmadd_ps` / `_mm256_mul_ps`).
  - [ ] In-memory contiguous cache-aligned vector storage (avoiding random heap lookups).
  - [ ] Sub-millisecond retrieval latency (< 1.0ms query-to-document retrieval time).

- [ ] **Index & Knowledge Base Ingestion**
  - [ ] Ingest operational markdown manuals:
    - [ ] `manuals/atc_emergency_manual.md` (Air traffic altitude, separation, transponder squawk codes).
    - [ ] Chemical plant emergency overpressure tolerances (MAWP, venting threshold, quench rates).
  - [ ] Chunking strategy tailored for technical manuals (section-aware, metric-aware).
  - [ ] Dense embeddings pre-computed and stored in memory for immediate access.
  - [ ] Extensible corpus ingestion pipeline (capable of adding FAA flight manuals, OSHA chemical safety standards).

---

## 4. 🛡️ Deterministic Post-Retrieval Validation (Anti-Hallucination Guardrail)

- [ ] **Zero-Hallucination Numeric Enforcement**
  - [ ] Strict deterministic AST / Regex parsing on candidate output tokens before audio synthesis.
  - [ ] Extract all numerical entities (altitudes, squawk codes, pressures, flow rates) from generated text.
  - [ ] Cross-validate numerical tokens against the retrieved source manual chunks.
  - [ ] Deterministic replacement / fallback rule: If a generated numeric value deviates from retrieved ground truth, enforce strict ground-truth override.

- [ ] **Ground Truth Benchmark Compliance (`ground_truth_numeric_validation.json`)**
  - [ ] **Query 1 (`query_01_atc_squawk.wav`):**
    - [ ] Intent: `transponder_squawk_code`
    - [ ] Ground Truth Numeric: Strictly validate **`7700`** (General Emergency).
    - [ ] Negative check: Guarantee non-hallucination of 7500 (Hijack) or 7600 (NORDO) unless context dictates.
  - [ ] **Query 2 (`query_02_reactor_pressure.wav`):**
    - [ ] Intent: `reactor_mawp_threshold`
    - [ ] Ground Truth Numerics: Strictly validate **`450.0`** (MAWP) and **`485.5`** (Venting trip threshold).
    - [ ] Negative check: Ensure no rounding errors or fabricated pressure units.

---

## 5. ⏱️ Latency & Production Performance Criteria

- [ ] **Sub-250ms Response Latency Constraint**
  - [ ] Measure exact time elapsed from **User Speech End (VAD Speech-Off)** to **First Synthesized Audio Chunk Egress (TTFA)**.
  - [ ] Strict threshold: $\text{Latency}_{\text{End-of-Speech} \to \text{TTFA}} < \mathbf{250\text{ ms}}$.
  - [ ] Maintain consistent sub-250ms performance across both benchmark queries.

- [ ] **Real-Time Telemetry & Profiling Dashboard**
  - [ ] Detailed micro-benchmarking breakdown for each stage:
    1. Ingestion & Ring Buffer delay ($T_{\text{buf}}$)
    2. Early Intent Prediction time ($T_{\text{intent}}$)
    3. SIMD Vector Retrieval latency ($T_{\text{simd}}$)
    4. Speculative Draft Validation time ($T_{\text{draft\_val}}$)
    5. Deterministic Numeric Verification latency ($T_{\text{numeric\_val}}$)
    6. Time To First Audio frame ($T_{\text{TTFA}}$)
  - [ ] Visual or console benchmark runner displaying exact millisecond timestamps and PASS/FAIL badge for the 250ms limit.

---

## 6. 🛠️ Mandatory Language & Toolchain Compliance

- [ ] **C++ Component:**
  - [ ] High-speed SIMD-vectorized vector search (AVX2).
  - [ ] Fast ring buffer / data structures.
  - [ ] Deterministic validator / string parser.

- [ ] **Networking Component (Go / Concurrent WebSocket layer):**
  - [ ] Concurrent streaming WebSocket / IPC server for low-overhead audio transport.

- [ ] **Python Component:**
  - [ ] Acoustic feature processing & speech models.
  - [ ] Pipeline orchestration & benchmark evaluation runner.

---

## 7. 🧪 Testing, Benchmark & Demo Deliverables

- [ ] **Automated Benchmark Runner (`evaluate_benchmark.py` / CLI tool):**
  - [ ] Feeds `query_01_atc_squawk.wav` and `query_02_reactor_pressure.wav` frame-by-frame simulating live microphone stream.
  - [ ] Evaluates latency against the strict 250ms ceiling.
  - [ ] Verifies output text and audio against `ground_truth_numeric_validation.json`.
  - [ ] Generates compliance report (Latency, Intent Match, Numeric Guardrail Integrity).

- [ ] **Interactive Demonstration Interface:**
  - [ ] Live audio streaming demo (microphone or file injection).
  - [ ] Live visualization of the speculative retrieval triggering *before* audio finishes.
  - [ ] Instant audio playback of the response.
