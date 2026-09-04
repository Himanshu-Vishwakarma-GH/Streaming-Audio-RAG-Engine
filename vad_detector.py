"""
=============================================================================
Silero VAD (ONNX) Real-Time Streaming Voice Activity Detector for PS-003
=============================================================================
Features:
1. Powered by official Silero VAD v5 ONNX model (models/silero_vad.onnx).
2. Streaming frame-by-frame inference on 16kHz mono audio.
3. Stateful recurrent context tracking via [2, 1, 128] hidden state tensor.
4. Aggressive low-latency silence endpointing configurable to 80ms - 100ms.
5. Emits SPEECH_START, SPEECH_ACTIVE, and SPEECH_END events with millisecond timestamps.
=============================================================================
"""

import os
import time
import numpy as np
import onnxruntime as ort

class SileroVADDetector:
    def __init__(
        self,
        model_path: str = "models/silero_vad.onnx",
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 100,
        silence_timeout_ms: int = 80,  # Strict low-latency endpointing for PS-003
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Silero VAD ONNX model not found at {model_path}")
        
        # Configure ONNX runtime session with 1 thread for lowest context-switch latency
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.silence_timeout_ms = silence_timeout_ms
        self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        
        # Internal Silero VAD state [2, 1, 128]
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._rolling_context = np.zeros(64, dtype=np.float32)  # Silero v5 requires 64 rolling context samples
        self._buffer = np.zeros(0, dtype=np.float32)            # Ingestion buffer
        
        # Turn-taking state tracking
        self.is_speech_active = False
        self.speech_start_time_ms = 0.0
        self.speech_start_sample_idx = 0
        self.last_speech_time_ms = 0.0
        self.total_samples_processed = 0
        self.consecutive_silence_ms = 0.0
        self.consecutive_speech_ms = 0.0
        self.last_prob = 0.0
        
    def reset(self):
        """Reset internal recurrent state and turn tracking."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._rolling_context = np.zeros(64, dtype=np.float32)
        self._buffer = np.zeros(0, dtype=np.float32)
        self.is_speech_active = False
        self.speech_start_time_ms = 0.0
        self.speech_start_sample_idx = 0
        self.last_speech_time_ms = 0.0
        self.total_samples_processed = 0
        self.consecutive_silence_ms = 0.0
        self.consecutive_speech_ms = 0.0
        self.last_prob = 0.0

    def process_pcm_frame(self, pcm_bytes: bytes, frame_duration_ms: float = 20.0):
        """
        Process a single 16-bit 16kHz PCM audio chunk (e.g. 20ms = 320 samples = 640 bytes).
        """
        # Convert 16-bit PCM bytes to float32 in [-1.0, 1.0]
        samples_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        
        num_samples = len(samples_float32)
        self.total_samples_processed += num_samples
        current_time_ms = (self.total_samples_processed / self.sample_rate) * 1000.0
        
        # Append to buffer
        self._buffer = np.concatenate([self._buffer, samples_float32])
        
        # Silero V5 requires 576 samples (64 context + 512 window)
        speech_prob = self.last_prob
        while len(self._buffer) >= 512:
            chunk_512 = self._buffer[:512]
            self._buffer = self._buffer[512:]
            
            eval_window = np.concatenate([self._rolling_context, chunk_512]).reshape(1, 576).astype(np.float32)
            self._rolling_context = eval_window[0, -64:].copy()
            
            ort_inputs = {
                "input": eval_window,
                "state": self._state,
                "sr": self._sr_tensor
            }
            ort_outs = self.session.run(None, ort_inputs)
            speech_prob = float(ort_outs[0][0, 0])
            self._state = ort_outs[1]
            self.last_prob = speech_prob
        
        is_speech = speech_prob >= self.threshold
        event = None
        speech_duration_ms = 0.0
        
        if is_speech:
            self.consecutive_speech_ms += frame_duration_ms
            self.consecutive_silence_ms = 0.0
            
            if not self.is_speech_active:
                if self.consecutive_speech_ms >= self.min_speech_duration_ms:
                    self.is_speech_active = True
                    self.speech_start_time_ms = current_time_ms - self.consecutive_speech_ms
                    self.speech_start_sample_idx = self.total_samples_processed - int(self.consecutive_speech_ms * self.sample_rate / 1000.0)
                    event = "SPEECH_START"
            else:
                event = "SPEECH_ACTIVE"
                speech_duration_ms = current_time_ms - self.speech_start_time_ms
            self.last_speech_time_ms = current_time_ms
        else:
            self.consecutive_silence_ms += frame_duration_ms
            self.consecutive_speech_ms = 0.0
            
            if self.is_speech_active:
                speech_duration_ms = current_time_ms - self.speech_start_time_ms
                if self.consecutive_silence_ms >= self.silence_timeout_ms:
                    self.is_speech_active = False
                    event = "SPEECH_END"
                else:
                    event = "SPEECH_ACTIVE"
        
        return {
            "speech_prob": speech_prob,
            "is_speech": is_speech,
            "is_speech_active": self.is_speech_active,
            "event": event,
            "speech_duration_ms": speech_duration_ms,
            "timestamp_ms": current_time_ms,
        }

if __name__ == "__main__":
    import wave
    test_wav = "audio_samples/query_01_atc_squawk.wav"
    print(f"Testing SileroVADDetector on {test_wav}...")
    
    vad = SileroVADDetector()
    with wave.open(test_wav, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_pcm = wf.readframes(n_frames)
    
    chunk_size = 320 * 2  # 20ms @ 16kHz 16-bit = 640 bytes
    events = []
    
    t0 = time.perf_counter()
    for offset in range(0, len(raw_pcm), chunk_size):
        chunk = raw_pcm[offset:offset + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk + b"\x00" * (chunk_size - len(chunk))
        res = vad.process_pcm_frame(chunk, frame_duration_ms=20.0)
        if res["event"]:
            events.append((res["timestamp_ms"], res["event"], res["speech_prob"]))
    t1 = time.perf_counter()
    
    total_audio_ms = (n_frames / framerate) * 1000.0
    inference_ms = (t1 - t0) * 1000.0
    print(f"Audio Duration: {total_audio_ms:.1f}ms | Total VAD Processing: {inference_ms:.2f}ms")
    print(f"Average Frame Latency: {inference_ms / (len(raw_pcm) / chunk_size):.3f}ms (<1.0ms target)")
    print("\nDetected Events:")
    for ts, ev, p in events:
        print(f"  [{ts:7.1f} ms] Event: {ev:14s} (prob: {p:.3f})")
