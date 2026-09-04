# Project Persistent Memory & Context Tracker
## PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine

> **Document Purpose:** Self-updating persistent agent memory. Tracks current status, environment specs, architectural decisions, ground-truth constraints, and phase execution progress across the entire hackathon lifecycle.

---

## 1. 📌 Executive Context & Core Mission

* **Hackathon:** Intense 5-Hour National Hackathon
* **Problem Statement:** PS-003 — Speculative-Decoded Sub-250ms Streaming Audio RAG Engine
* **Track Domain:** Streaming Multimedia / Speech Processing / Low-Latency RAG
* **Core Mandate:**
  1. Round-trip response latency strictly **< 250 milliseconds** (from user speech end / VAD cutoff to first synthesized audio packet egress).
  2. **100% Deterministic Numeric Guardrail** (zero hallucinations on squawk codes, altitudes, pressures).
  3. **Mandatory Languages:** C++ (SIMD vector search & ring buffers), Go (concurrent WebSocket networking), Python (acoustics, models, evaluation).
* **Repository Remote:** `https://github.com/Himanshu-Vishwakarma-GH/Streaming-Audio-RAG-Engine.git` (Branch: `main`)

---

## 2. 💻 System & Hardware Profile

* **Host OS:** Windows (PowerShell)
* **Processor (CPU):** AMD Ryzen 7 5700U (8 physical cores, 16 logical threads, 4.3 GHz boost)
  * **Hardware Vector Extensions:** AVX, AVX2, FMA3 (Capable of 16 single-precision FLOPs per clock cycle)
* **C++ Compiler:** MSYS2 MinGW-w64 `g++.exe` 14.2.0 (Supports `-mavx2 -mfma -O3 -shared`)
* **Python Environments:**
  * Active Runtime: `C:\Users\Himanshu\AppData\Local\Programs\Python\Python310\python.exe`
  * Installed Packages: `torch` 2.9.1+cpu, `transformers`, `sentence-transformers`, `soundfile`, `sounddevice`, `onnxruntime`, `fastapi`, `uvicorn`, `websockets`, `scipy`, `numpy`
* **Node.js:** v24.0.0
* **Go Compiler:** Can be invoked / targeted via Go toolchain for networking server.

### 🌐 Open-Source Dependency Links (Always Available in Memory):
* **[snakers4/silero-vad](https://github.com/snakers4/silero-vad):** Pre-trained enterprise VAD (<1ms frame inference).
* **[rhasspy/piper](https://github.com/rhasspy/piper):** Fast offline neural TTS (VITS ONNX architecture).
* **[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):** Unified Next-Gen Kaldi offline speech runtime.
* **[ashvardanian/simsimd](https://github.com/ashvardanian/simsimd):** SIMD vector distance kernels (AVX2/AVX-512).
* **[unum-cloud/usearch](https://github.com/unum-cloud/usearch):** Minimalist vector search engine.
* **[cameron314/readerwriterqueue](https://github.com/cameron314/readerwriterqueue):** Lock-free SPSC circular queue for C++.
* **[coder/websocket](https://github.com/coder/websocket):** Ultra-low-allocation Go WebSocket library.
* **[UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers):** Fast dense vector embedding models.
* **[SalesforceAIResearch/VoiceAgentRAG](https://github.com/SalesforceAIResearch/VoiceAgentRAG):** Dual-agent speculative voice caching.
* **[NVIDIA/voice-agent-examples](https://github.com/NVIDIA/voice-agent-examples):** Speculative speech processing and TTS response cacher.

---

## 3. 🎯 Benchmark Ground Truth & Verification Targets

From `ground_truth_numeric_validation.json` and `manuals/atc_emergency_manual.md`:

| Query ID | Audio File | Target Intent | Ground Truth Numeric | Manual Source Section | Anti-Hallucination Guardrail Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **q1** | `query_01_atc_squawk.wav` (1.5s, 16kHz PCM) | `transponder_squawk_code` | **`7700`** | Section 1: Transponder Squawk codes | Must return General Emergency squawk **`7700`**. Strictly block 7500 (Hijack) and 7600 (NORDO). |
| **q2** | `query_02_reactor_pressure.wav` (1.5s, 16kHz PCM) | `reactor_mawp_threshold` | **`450.0`**, **`485.5`** | Section 2: Chemical Plant Overpressure | Must return Core R-101 MAWP = **`450.0 PSI`** and Emergency Venting Trip = **`485.5 PSI`**. |

---

## 4. 🚀 Phase Execution Tracker

| Phase | Description | Estimated Time | Status | Artifacts / Deliverables |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | **C++ AVX2 SIMD Engine & Manual Indexer** | 45 mins | 🟡 **CURRENT (Ready to Begin)** | `simd_engine.cpp`, `simd_engine.dll`, `manual_indexer.py`, `simd_wrapper.py` |
| **Phase 2** | **Streaming Ingestion & Lock-Free Ring Buffer** | 40 mins | ⚪ Pending | `streaming_ws.go`, `audio_stream_handler.py`, `vad_detector.py` |
| **Phase 3** | **Early Intent Detector & Speculative Trigger** | 45 mins | ⚪ Pending | `early_intent_detector.py`, `speculative_engine.py` |
| **Phase 4** | **Deterministic Post-Retrieval Guardrail** | 30 mins | ⚪ Pending | `numeric_guardrail.py` |
| **Phase 5** | **Speculative Neural Vocoder / TTS & Cache** | 40 mins | ⚪ Pending | `tts_streamer.py`, `response_cacher.py` |
| **Phase 6** | **Automated Benchmark Suite & Demo UI** | 30 mins | ⚪ Pending | `evaluate_benchmark.py`, `demo_app.py`, `BENCHMARK_RESULTS.md` |

---

## 5. 🧠 Critical Architectural Decisions & Formulas

1. **Why Speculative Execution is Mandatory:**
   - Sequential execution takes $\approx 1.8\text{s} - 4.0\text{s}$, mathematically failing the 250ms target.
   - Using the **Tool-Intent Stabilization formula**:
     $$H = \min(L, \, (N - k^*) \cdot \delta)$$
     Detecting intent at prefix $k^* = 450\text{ms}$ into a $1,500\text{ms}$ utterance hides 100% of retrieval ($L < 0.02\text{ms}$), guardrail checking ($0.3\text{ms}$), and first-frame TTS synthesis ($35\text{ms}$) behind user speech.
2. **Post-Speech Latency Math:**
   $$T_{\text{post-speech}} = T_{\text{silence\_VAD}} (80\text{ms}) + T_{\text{commit\_check}} (2\text{ms}) + T_{\text{wire\_flush}} (5\text{ms}) \approx \mathbf{87\text{ ms}} \ll \mathbf{250\text{ ms}}$$
3. **Why Flat AVX2 Scan Beats HNSW Graph for Hackathon Corpus:**
   - For 20–100 technical manual chunks, flat scan with `_mm256_fmadd_ps` takes **< 0.02 milliseconds**, has zero graph-building overhead, zero approximate recall error, and executes directly in CPU L1/L2 cache.
4. **Why Monolithic Audio LLMs (Moshi/Mini-Omni) Are Excluded:**
   - They hallucinate numeric data, cannot enforce AST regex guardrails, require 24GB VRAM, and cannot meet deterministic verification.

---

## 6. 📝 Chronological Changelog & Updates

* **[11:08 - 11:20]**: Initialized project, parsed `Problem Statement.md`, inspected benchmark WAV files and ground truth JSON. Checked toolchain (g++ 14.2.0, Python 3.10 with Torch, Node v24).
* **[11:20 - 11:40]**: Created `PS Objective.md` detailed requirements checklist.
* **[11:40 - 12:00]**: Dispatched 3 parallel research subagents. Synthesized findings into `OpenSource-SimilarReport.md` and created `Recommended Architecture & Pipeline.md`.
* **[12:00 - 12:08]**: Produced final strategic documents: `Final Plan.md`, `Architecture.md`, and `Phases.md`.
* **[12:08 - 12:12]**: Initialized Git repository, configured `.gitignore`, generated `README.md`, and pushed initial commit to GitHub remote `origin main`.
* **[12:12 - Current]**: Created `memory.md` to permanently store execution state, hardware profile, and phase milestones. Ready to launch **Phase 1**.
