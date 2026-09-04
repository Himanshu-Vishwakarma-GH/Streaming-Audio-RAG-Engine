# PS-003: Speculative-Decoded Sub-250ms Streaming Audio RAG Engine

> **Track Domain:** Streaming Multimedia / Speech Processing / Low-Latency RAG

---

## 📌 Scenario & Technical Challenge
High-stakes operations (such as air traffic control, emergency dispatch, and industrial plant control) require instant hands-free voice assistance. Operators need to speak complex technical questions and hear immediate, accurate spoken answers. Teams must engineer an end-to-end, bidirectional streaming audio RAG system that processes live audio chunks and begins streaming synthesized speech back before the user has even finished their sentence.

## 🚨 Production / Industry Bottleneck
Conventional voice pipelines execute steps sequentially: *Speech-to-Text -> Text Embedding -> Vector Search -> LLM Generation -> Text-to-Speech*. This cascade introduces a compounding latency delay of **1.8 to 4.0 seconds**, making conversational voice systems dangerously slow during time-critical operational emergencies.

## 💡 Desired Solution & Technical Hints
Decouple the pipeline using speculative execution. Ingest continuous PCM audio streams via WebSockets or WebRTC. Predict user intent and extract keywords from early phoneme fragments before full sentence completion. Speculatively query an in-memory vector index (e.g., quantized HNSW) while the user is still speaking, pipelining draft response tokens directly into a streaming neural vocoder.

## 🛠️ Mandatory Languages
C++ or Rust (SIMD-accelerated vector search and ring buffers), Go (Concurrent WebSocket/WebRTC networking layer), Python (Acoustic and speech models).

## 🎯 Production Criteria
Total round-trip response latency (from the moment the user stops speaking to the first synthesized audio output) must remain strictly **under 250 milliseconds**, with deterministic post-retrieval validation ensuring critical numerical data (altitudes, coordinates, pressures) are never hallucinated.

---

## 📦 Dataset Package & Ingestion Policy

### Provided in `PS-003.zip`:
- `manuals/`: Operational markdown manuals covering Air Traffic Control protocols (radar separation, squawk codes, emergency descent) and Chemical Plant reactor pressure relief limits.
- `audio_samples/`: Sample 16kHz PCM audio files of urgent voice queries.
- `ground_truth_numeric_validation.json`: Expected numerical ground truth to evaluate anti-hallucination guardrails.
- `DATASET_INFO.md`: Schema and query specifications.

### 🌐 External Data & Ingestion Liberty:
Participants have complete liberty to ingest additional technical manuals (e.g., **FAA Flight Procedures Manuals**, **OSHA Chemical Safety Standards**) and public air traffic audio archives (e.g., **LiveATC.net**) to broaden retrieval coverage and acoustic variety.

---

### How to Extract:
```bash
# Navigate to this folder and extract the dataset
unzip PS-003.zip -d dataset/
```