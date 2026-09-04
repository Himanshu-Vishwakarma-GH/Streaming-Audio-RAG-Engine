package main

import (
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// RFC 6455 WebSocket Magic GUID
const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// 16kHz 16-bit Mono PCM: 20ms frame = 320 samples = 640 bytes
const (
	FrameDurationMs = 20
	SampleRate      = 16000
	FrameBytes      = 640
)

// ServerMetrics holds real-time telemetry
type ServerMetrics struct {
	TotalConnections int64   `json:"total_connections"`
	ActiveConnections int64   `json:"active_connections"`
	TotalFrames      int64   `json:"total_frames"`
	TotalBytes       int64   `json:"total_bytes"`
	AvgFrameTimeUs   float64 `json:"avg_frame_time_us"`
}

var (
	metrics ServerMetrics
	mu      sync.Mutex
)

// FramePool to eliminate memory allocations during high-frequency 50Hz audio streaming
var framePool = sync.Pool{
	New: func() interface{} {
		b := make([]byte, FrameBytes)
		return &b
	},
}

// performHandshake upgrades HTTP GET to RFC 6455 WebSocket
func performHandshake(w http.ResponseWriter, r *http.Request) (net.Conn, error) {
	if r.Header.Get("Upgrade") != "websocket" {
		http.Error(w, "Expected websocket upgrade", http.StatusBadRequest)
		return nil, fmt.Errorf("not a websocket handshake")
	}

	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		http.Error(w, "Missing Sec-WebSocket-Key", http.StatusBadRequest)
		return nil, fmt.Errorf("missing Sec-WebSocket-Key")
	}

	h := sha1.New()
	h.Write([]byte(key + wsGUID))
	acceptKey := base64.StdEncoding.EncodeToString(h.Sum(nil))

	hj, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "Hijacking not supported", http.StatusInternalServerError)
		return nil, fmt.Errorf("hijacking failed")
	}

	conn, bufrw, err := hj.Hijack()
	if err != nil {
		return nil, err
	}

	response := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: " + acceptKey + "\r\n\r\n"

	if _, err := bufrw.WriteString(response); err != nil {
		conn.Close()
		return nil, err
	}
	if err := bufrw.Flush(); err != nil {
		conn.Close()
		return nil, err
	}

	return conn, nil
}

// readFrame reads and decodes an RFC 6455 WebSocket frame
func readFrame(r io.Reader) (opcode byte, payload []byte, err error) {
	var header [2]byte
	if _, err = io.ReadFull(r, header[:]); err != nil {
		return 0, nil, err
	}

	opcode = header[0] & 0x0F
	masked := (header[1] & 0x80) != 0
	payloadLen := int64(header[1] & 0x7F)

	if payloadLen == 126 {
		var ext [2]byte
		if _, err = io.ReadFull(r, ext[:]); err != nil {
			return 0, nil, err
		}
		payloadLen = int64(binary.BigEndian.Uint16(ext[:]))
	} else if payloadLen == 127 {
		var ext [8]byte
		if _, err = io.ReadFull(r, ext[:]); err != nil {
			return 0, nil, err
		}
		payloadLen = int64(binary.BigEndian.Uint64(ext[:]))
	}

	var mask [4]byte
	if masked {
		if _, err = io.ReadFull(r, mask[:]); err != nil {
			return 0, nil, err
		}
	}

	payload = make([]byte, payloadLen)
	if _, err = io.ReadFull(r, payload); err != nil {
		return 0, nil, err
	}

	if masked {
		for i := int64(0); i < payloadLen; i++ {
			payload[i] ^= mask[i%4]
		}
	}

	return opcode, payload, nil
}

// writeBinaryFrame sends an unmasked binary WebSocket frame back to client
func writeBinaryFrame(w io.Writer, payload []byte) error {
	length := len(payload)
	var header []byte

	if length <= 125 {
		header = []byte{0x82, byte(length)}
	} else if length <= 65535 {
		header = []byte{0x82, 126, 0, 0}
		binary.BigEndian.PutUint16(header[2:], uint16(length))
	} else {
		header = []byte{0x82, 127, 0, 0, 0, 0, 0, 0, 0, 0}
		binary.BigEndian.PutUint64(header[2:], uint64(length))
	}

	if _, err := w.Write(header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

// wsAudioHandler processes live 16kHz PCM streaming connection
func wsAudioHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := performHandshake(w, r)
	if err != nil {
		log.Printf("[WebSocket] Handshake failed: %v", err)
		return
	}
	defer conn.Close()

	atomic.AddInt64(&metrics.TotalConnections, 1)
	atomic.AddInt64(&metrics.ActiveConnections, 1)
	defer atomic.AddInt64(&metrics.ActiveConnections, -1)

	clientAddr := conn.RemoteAddr().String()
	log.Printf("[WebSocket] Client connected from %s", clientAddr)

	var frameCount int64
	var byteCount int64
	t0 := time.Now()

	for {
		fStart := time.Now()
		opcode, payload, err := readFrame(conn)
		if err != nil {
			if err != io.EOF {
				log.Printf("[WebSocket] Read error from %s: %v", clientAddr, err)
			}
			break
		}

		// 0x8 = Connection Close, 0x9 = Ping, 0xA = Pong
		if opcode == 0x8 {
			log.Printf("[WebSocket] Client sent close frame: %s", clientAddr)
			break
		} else if opcode == 0x9 {
			// Respond with Pong (0xA)
			conn.Write([]byte{0x8A, 0x00})
			continue
		}

		// 0x2 = Binary Frame (16kHz PCM audio chunk)
		if opcode == 0x2 || opcode == 0x1 {
			chunkLen := len(payload)
			frameCount++
			byteCount += int64(chunkLen)

			atomic.AddInt64(&metrics.TotalFrames, 1)
			atomic.AddInt64(&metrics.TotalBytes, int64(chunkLen))

			fDurationUs := float64(time.Since(fStart).Microseconds())
			mu.Lock()
			if metrics.AvgFrameTimeUs == 0 {
				metrics.AvgFrameTimeUs = fDurationUs
			} else {
				metrics.AvgFrameTimeUs = (metrics.AvgFrameTimeUs * 0.95) + (fDurationUs * 0.05)
			}
			mu.Unlock()

			// Echo frame back for round-trip duplex telemetry
			if err := writeBinaryFrame(conn, payload); err != nil {
				log.Printf("[WebSocket] Write error to %s: %v", clientAddr, err)
				break
			}
		}
	}

	elapsedSec := time.Since(t0).Seconds()
	log.Printf("[WebSocket] Session closed for %s: %d frames (%d bytes) in %.2fs (%.1f frames/sec)",
		clientAddr, frameCount, byteCount, elapsedSec, float64(frameCount)/elapsedSec)
}

func main() {
	port := 8765

	http.HandleFunc("/ws/audio", wsAudioHandler)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "healthy",
			"service": "Streaming Audio RAG Engine (Go WebSocket)",
			"sample_rate": SampleRate,
			"frame_ms": FrameDurationMs,
		})
	})
	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		mu.Lock()
		defer mu.Unlock()
		json.NewEncoder(w).Encode(metrics)
	})

	addr := fmt.Sprintf("0.0.0.0:%d", port)
	log.Printf("==========================================================")
	log.Printf("🚀 PS-003 Go Streaming Audio WebSocket Server listening on %s", addr)
	log.Printf("Endpoints: ws://localhost:%d/ws/audio | http://localhost:%d/health", port, port)
	log.Printf("==========================================================")

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Fatal server error: %v", err)
	}
}
