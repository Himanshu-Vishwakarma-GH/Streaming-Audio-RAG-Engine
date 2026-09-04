"""
=============================================================================
Streaming Audio Ingestion Handler & C++ Lock-Free Ring Buffer Bridge (PS-003)
=============================================================================
Responsibilities:
1. Ingests raw 16kHz 16-bit PCM audio chunks (20ms = 320 samples = 640 bytes).
2. Pushes frames directly into the C++ SIMD lock-free SPSC circular ring buffer
   via `simd_wrapper.py` with zero heap reallocations.
3. Chains each frame through Silero VAD (ONNX) to track real-time speech energy
   and trigger sub-100ms silence endpointing.
4. Emits streaming callbacks:
   - `on_chunk(pcm_bytes, vad_res)`: Fired for every 20ms frame.
   - `on_early_prefix(accumulated_pcm, duration_ms)`: Fired continuously during speech.
   - `on_speech_end(total_duration_ms, full_utterance_pcm)`: Fired on VAD Speech-Off.
=============================================================================
"""

import time
import numpy as np
from typing import Callable, Optional
from simd_wrapper import AudioRingBuffer
from vad_detector import SileroVADDetector

class AudioStreamHandler:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: float = 20.0,
        ring_buffer_capacity: int = 1024,
        silence_timeout_ms: int = 80,
        vad_threshold: float = 0.5,
        on_chunk: Optional[Callable] = None,
        on_early_prefix: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.bytes_per_sample = 2  # 16-bit PCM
        self.samples_per_frame = int(sample_rate * (frame_duration_ms / 1000.0))  # 320 samples
        self.bytes_per_frame = self.samples_per_frame * self.bytes_per_sample     # 640 bytes
        
        # 1. Initialize C++ Lock-Free SPSC Ring Buffer
        self.ring_buf = AudioRingBuffer(capacity_frames=ring_buffer_capacity)
        
        # 2. Initialize Neural Silero VAD
        self.vad = SileroVADDetector(
            sample_rate=sample_rate,
            threshold=vad_threshold,
            silence_timeout_ms=silence_timeout_ms,
        )
        
        # Callbacks
        self.on_chunk = on_chunk
        self.on_early_prefix = on_early_prefix
        self.on_speech_end = on_speech_end
        
        # Active streaming state
        self.stream_buffer = bytearray()
        self.utterance_pcm = bytearray()
        self.total_frames_ingested = 0
        self.total_bytes_ingested = 0
        self.speech_active = False
        self.speech_start_time_ms = 0.0

    def reset(self):
        """Reset internal buffers and VAD state."""
        self.ring_buf.clear()
        self.vad.reset()
        self.stream_buffer.clear()
        self.utterance_pcm.clear()
        self.total_frames_ingested = 0
        self.total_bytes_ingested = 0
        self.speech_active = False
        self.speech_start_time_ms = 0.0

    def push_audio_chunk(self, raw_bytes: bytes):
        """
        Ingest incoming byte chunk (can be any size, e.g. from WebSocket).
        Buffers bytes and extracts complete 20ms frames (640 bytes).
        """
        self.stream_buffer.extend(raw_bytes)
        self.total_bytes_ingested += len(raw_bytes)
        
        results = []
        while len(self.stream_buffer) >= self.bytes_per_frame:
            frame_bytes = bytes(self.stream_buffer[:self.bytes_per_frame])
            del self.stream_buffer[:self.bytes_per_frame]
            
            res = self._process_frame(frame_bytes)
            results.append(res)
            
        return results

    def _process_frame(self, frame_bytes: bytes):
        """Process one exact 20ms frame."""
        self.total_frames_ingested += 1
        elapsed_ms = self.total_frames_ingested * self.frame_duration_ms
        
        # 1. Push into C++ Lock-Free Ring Buffer (lossless & non-blocking)
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        pushed = self.ring_buf.push_frame(samples)
        
        # 2. Track speech with Silero VAD
        vad_res = self.vad.process_pcm_frame(frame_bytes, frame_duration_ms=self.frame_duration_ms)
        event = vad_res["event"]
        
        if event == "SPEECH_START":
            self.speech_active = True
            self.utterance_pcm.clear()
            self.utterance_pcm.extend(frame_bytes)
            self.speech_start_time_ms = elapsed_ms
        elif event == "SPEECH_ACTIVE":
            if self.speech_active:
                self.utterance_pcm.extend(frame_bytes)
                speech_duration = elapsed_ms - self.speech_start_time_ms
                if self.on_early_prefix:
                    self.on_early_prefix(bytes(self.utterance_pcm), speech_duration)
        elif event == "SPEECH_END":
            if self.speech_active:
                self.speech_active = False
                total_duration = elapsed_ms - self.speech_start_time_ms
                if self.on_speech_end:
                    self.on_speech_end(total_duration, bytes(self.utterance_pcm))
        
        if self.on_chunk:
            self.on_chunk(frame_bytes, vad_res)
            
        return {
            "frame_index": self.total_frames_ingested,
            "elapsed_ms": elapsed_ms,
            "ring_buf_pushed": pushed,
            "vad": vad_res,
        }

if __name__ == "__main__":
    import wave
    test_file = "audio_samples/query_01_atc_squawk.wav"
    print(f"Testing AudioStreamHandler on {test_file}...")
    
    events_log = []
    def handle_early_prefix(pcm, dur_ms):
        if int(dur_ms) in [400, 600, 800, 1000]:
            events_log.append(f"Early prefix at {dur_ms:.0f}ms (PCM size: {len(pcm)} bytes)")
            
    def handle_speech_end(dur_ms, pcm):
        events_log.append(f"Speech END detected at {dur_ms:.0f}ms (Total PCM: {len(pcm)} bytes)")
    
    handler = AudioStreamHandler(
        silence_timeout_ms=80,
        on_early_prefix=handle_early_prefix,
        on_speech_end=handle_speech_end,
    )
    
    with wave.open(test_file, "rb") as wf:
        raw_pcm = wf.readframes(wf.getnframes())
        
    t0 = time.perf_counter()
    # Simulate streaming network packets of 640 bytes (20ms)
    chunk_size = 640
    frame_latencies = []
    for i in range(0, len(raw_pcm), chunk_size):
        chunk = raw_pcm[i:i + chunk_size]
        f_t0 = time.perf_counter()
        handler.push_audio_chunk(chunk)
        frame_latencies.append((time.perf_counter() - f_t0) * 1000.0)
        
    t1 = time.perf_counter()
    
    print(f"Ingested {handler.total_frames_ingested} frames ({handler.total_bytes_ingested} bytes).")
    print(f"Ring Buffer Available for Read: {handler.ring_buf.available_read()} frames.")
    print(f"Average Frame Ingestion Latency: {np.mean(frame_latencies):.3f}ms (< 1.0ms target)")
    print(f"p95 Frame Latency: {np.percentile(frame_latencies, 95):.3f}ms")
    print("Events captured during streaming:")
    for ev in events_log:
        print("  *", ev)
