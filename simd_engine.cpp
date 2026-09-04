/**
 * ============================================================================
 * SIMD Engine for PS-003: Ultra-Low-Latency Audio RAG Pipeline
 * ============================================================================
 * 
 * Features:
 * 1. 32-byte cache-aligned vector matrix storage for 384-dimensional embeddings.
 * 2. Unrolled AVX2 FMA dot-product kernel (_mm256_fmadd_ps) with dual accumulators
 *    for maximum throughput and sub-microsecond flat index search.
 * 3. Flat top-1 vector similarity search returning best chunk index and dot-product score.
 * 4. Lock-free Single-Producer Single-Consumer (SPSC) circular ring buffer with
 *    alignas(64) cache-line padded atomic head and tail pointers to eliminate false
 *    sharing for 16kHz PCM audio frames (320 samples = 640 bytes per 20ms chunk).
 * 5. Full C-ABI export (__declspec(dllexport)) for zero-overhead invocation from
 *    Python (ctypes) and Go (cgo).
 * ============================================================================
 */

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <atomic>
#include <new>

#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__)
    #include <malloc.h>
    #define ALIGNED_ALLOC(size, align) _aligned_malloc((size), (align))
    #define ALIGNED_FREE(ptr)          _aligned_free((ptr))
#else
    #include <cstdlib>
    #define ALIGNED_ALLOC(size, align) aligned_alloc((align), (size))
    #define ALIGNED_FREE(ptr)          free((ptr))
#endif

#ifdef _WIN32
    #define SIMD_API __declspec(dllexport)
#else
    #define SIMD_API __attribute__((visibility("default")))
#endif

// ============================================================================
// 1. HARDWARE-ACCELERATED AVX2 FMA VECTOR KERNELS
// ============================================================================

/**
 * Horizontal summation helper: sums 8 single-precision floats in a 256-bit AVX register.
 */
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum128, sum128);
    sum128 = _mm_add_ps(sum128, shuf);
    shuf = _mm_shuffle_ps(sum128, sum128, 0x01);
    sum128 = _mm_add_ss(sum128, shuf);
    return _mm_cvtss_f32(sum128);
}

/**
 * Computes dot product of two vectors using unrolled AVX2 FMA (_mm256_fmadd_ps).
 * Employs two independent vector accumulators to saturate CPU execution ports
 * and hide FMA latency (4-5 cycles) for 384-dimensional dense vectors.
 */
extern "C" SIMD_API float avx2_dot_product(const float* a, const float* b, int dim) {
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();

    int i = 0;
    // Unrolled loop: processes 16 floats (64 bytes) per iteration
    for (; i <= dim - 16; i += 16) {
        __m256 va1 = _mm256_loadu_ps(a + i);
        __m256 vb1 = _mm256_loadu_ps(b + i);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 8);
        __m256 vb2 = _mm256_loadu_ps(b + i + 8);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
    }

    // Process remaining multiples of 8 floats
    for (; i <= dim - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc1 = _mm256_fmadd_ps(va, vb, acc1);
    }

    // Combine dual accumulators
    __m256 acc = _mm256_add_ps(acc1, acc2);
    float sum = hsum256_ps(acc);

    // Scalar tail for dimensions that are not multiples of 8
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

/**
 * C-ABI alias for avx2_dot_product
 */
extern "C" SIMD_API float simd_dot_product(const float* a, const float* b, int dim) {
    return avx2_dot_product(a, b, dim);
}

// ============================================================================
// 2. CACHE-ALIGNED VECTOR STORAGE & FLAT SEARCH
// ============================================================================

static float* g_vector_matrix = nullptr;
static int    g_num_vectors   = 0;
static int    g_dim           = 384;

/**
 * Allocates 32-byte cache-aligned memory for storing num_vectors of size dim.
 * Returns 0 on success, negative error code on failure.
 */
extern "C" SIMD_API int simd_init(int num_vectors, int dim) {
    if (num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    if (g_vector_matrix != nullptr) {
        ALIGNED_FREE(g_vector_matrix);
        g_vector_matrix = nullptr;
    }

    g_num_vectors = num_vectors;
    g_dim = dim;

    size_t total_bytes = static_cast<size_t>(num_vectors) * dim * sizeof(float);
    g_vector_matrix = static_cast<float*>(ALIGNED_ALLOC(total_bytes, 32));
    if (!g_vector_matrix) {
        g_num_vectors = 0;
        g_dim = 0;
        return -2;
    }

    std::memset(g_vector_matrix, 0, total_bytes);
    return 0;
}

/**
 * Loads raw vector data into the aligned memory matrix.
 * Allocates / re-allocates memory if necessary.
 * Returns 0 on success, negative error code on failure.
 */
extern "C" SIMD_API int simd_load_vectors(const float* data, int num_vectors, int dim) {
    if (!data || num_vectors <= 0 || dim <= 0) {
        return -1;
    }

    int rc = simd_init(num_vectors, dim);
    if (rc != 0) {
        return rc;
    }

    size_t total_bytes = static_cast<size_t>(num_vectors) * dim * sizeof(float);
    std::memcpy(g_vector_matrix, data, total_bytes);
    return 0;
}

/**
 * Frees vector memory and resets index state.
 */
extern "C" SIMD_API int simd_cleanup(void) {
    if (g_vector_matrix != nullptr) {
        ALIGNED_FREE(g_vector_matrix);
        g_vector_matrix = nullptr;
    }
    g_num_vectors = 0;
    g_dim = 0;
    return 0;
}

/**
 * Returns currently loaded number of vectors.
 */
extern "C" SIMD_API int simd_get_num_vectors(void) {
    return g_num_vectors;
}

/**
 * Returns vector dimension.
 */
extern "C" SIMD_API int simd_get_dim(void) {
    return g_dim;
}

/**
 * Performs flat exhaustive search against loaded vector matrix using AVX2 FMA.
 * Outputs the index of the closest vector (out_best_idx) and similarity score (out_best_score).
 * Returns 0 on success, negative error code on failure.
 */
extern "C" SIMD_API int simd_search_top1(const float* query_vec, int* out_best_idx, float* out_best_score) {
    if (!query_vec || !out_best_idx || !out_best_score) {
        return -1;
    }
    if (!g_vector_matrix || g_num_vectors <= 0 || g_dim <= 0) {
        *out_best_idx = -1;
        *out_best_score = -1.0f;
        return -2;
    }

    int best_idx = -1;
    float best_score = -1e30f;

    for (int i = 0; i < g_num_vectors; ++i) {
        const float* candidate = g_vector_matrix + (static_cast<size_t>(i) * g_dim);
        float score = avx2_dot_product(query_vec, candidate, g_dim);
        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    *out_best_idx = best_idx;
    *out_best_score = best_score;
    return 0;
}

// ============================================================================
// 3. SPSC LOCK-FREE CIRCULAR AUDIO RING BUFFER
// ============================================================================
// Frame specifications: 16kHz sample rate, 20ms duration
// 16000 * 0.020 = 320 samples. 16-bit PCM = 2 bytes/sample -> 640 bytes/frame.

constexpr size_t AUDIO_FRAME_SAMPLES = 320;
constexpr size_t AUDIO_FRAME_BYTES   = AUDIO_FRAME_SAMPLES * sizeof(int16_t);
constexpr size_t DEFAULT_RING_BUFFER_CAPACITY = 1024; // 1024 frames = ~20.48 sec

#pragma pack(push, 1)
struct AudioFrame {
    int16_t samples[AUDIO_FRAME_SAMPLES]; // Exactly 640 bytes
};
#pragma pack(pop)
static_assert(sizeof(AudioFrame) == 640, "AudioFrame must be exactly 640 bytes");

class SPSCAudioRingBuffer {
public:
    explicit SPSCAudioRingBuffer(size_t capacity_frames = DEFAULT_RING_BUFFER_CAPACITY) {
        // Enforce power-of-two capacity for fast bitwise masking
        m_capacity = 1;
        while (m_capacity < capacity_frames) {
            m_capacity <<= 1;
        }
        m_mask = m_capacity - 1;

        size_t total_bytes = m_capacity * sizeof(AudioFrame);
        m_frames = static_cast<AudioFrame*>(ALIGNED_ALLOC(total_bytes, 64));
        if (m_frames) {
            std::memset(m_frames, 0, total_bytes);
        }
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
    }

    ~SPSCAudioRingBuffer() {
        if (m_frames) {
            ALIGNED_FREE(m_frames);
            m_frames = nullptr;
        }
    }

    bool is_valid() const {
        return m_frames != nullptr;
    }

    size_t capacity() const {
        return m_capacity;
    }

    // Push 1 frame (320 samples / 640 bytes)
    bool push_frame(const int16_t* samples) {
        if (!m_frames || !samples) return false;

        const size_t head = m_head.load(std::memory_order_relaxed);
        const size_t tail = m_tail.load(std::memory_order_acquire);

        if ((head - tail) >= m_capacity) {
            return false; // Ring buffer full
        }

        std::memcpy(m_frames[head & m_mask].samples, samples, AUDIO_FRAME_BYTES);
        m_head.store(head + 1, std::memory_order_release);
        return true;
    }

    // Pop 1 frame (320 samples / 640 bytes)
    bool pop_frame(int16_t* out_samples) {
        if (!m_frames || !out_samples) return false;

        const size_t tail = m_tail.load(std::memory_order_relaxed);
        const size_t head = m_head.load(std::memory_order_acquire);

        if (tail == head) {
            return false; // Ring buffer empty
        }

        std::memcpy(out_samples, m_frames[tail & m_mask].samples, AUDIO_FRAME_BYTES);
        m_tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    size_t available_read() const {
        const size_t head = m_head.load(std::memory_order_acquire);
        const size_t tail = m_tail.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

    size_t available_write() const {
        const size_t head = m_head.load(std::memory_order_relaxed);
        const size_t tail = m_tail.load(std::memory_order_acquire);
        const size_t occupied = head - tail;
        return (occupied < m_capacity) ? (m_capacity - occupied) : 0;
    }

    void clear() {
        const size_t head = m_head.load(std::memory_order_relaxed);
        m_tail.store(head, std::memory_order_release);
    }

private:
    AudioFrame* m_frames = nullptr;
    size_t      m_capacity = 0;
    size_t      m_mask = 0;

    // Cache line isolation (64 bytes) to eliminate false sharing between producer and consumer cores
    alignas(64) std::atomic<size_t> m_head{0};
    alignas(64) std::atomic<size_t> m_tail{0};
};

static SPSCAudioRingBuffer* g_ring_buffer = nullptr;

/**
 * Initializes global SPSC audio ring buffer with specified capacity (in 20ms frames).
 */
extern "C" SIMD_API int ring_buffer_init(size_t capacity_frames) {
    if (g_ring_buffer) {
        delete g_ring_buffer;
        g_ring_buffer = nullptr;
    }
    g_ring_buffer = new (std::nothrow) SPSCAudioRingBuffer(capacity_frames > 0 ? capacity_frames : DEFAULT_RING_BUFFER_CAPACITY);
    if (!g_ring_buffer || !g_ring_buffer->is_valid()) {
        delete g_ring_buffer;
        g_ring_buffer = nullptr;
        return -1;
    }
    return 0;
}

/**
 * Pushes 1 frame of 320 int16 samples into the global ring buffer.
 * Returns 1 on success, 0 on buffer full or error.
 */
extern "C" SIMD_API int ring_buffer_push_frame(const int16_t* samples) {
    if (!g_ring_buffer) return 0;
    return g_ring_buffer->push_frame(samples) ? 1 : 0;
}

/**
 * Pushes multiple samples into the ring buffer in 320-sample (640-byte) frame chunks.
 * Returns number of complete frames successfully pushed.
 */
extern "C" SIMD_API int ring_buffer_push(const int16_t* samples, size_t num_samples) {
    if (!g_ring_buffer || !samples || num_samples < AUDIO_FRAME_SAMPLES) return 0;
    size_t frames = num_samples / AUDIO_FRAME_SAMPLES;
    size_t pushed = 0;
    for (size_t i = 0; i < frames; ++i) {
        if (g_ring_buffer->push_frame(samples + (i * AUDIO_FRAME_SAMPLES))) {
            pushed++;
        } else {
            break;
        }
    }
    return static_cast<int>(pushed);
}

/**
 * Pushes raw PCM bytes into the ring buffer in 640-byte frame increments.
 * Returns number of complete frames successfully pushed.
 */
extern "C" SIMD_API int ring_buffer_push_bytes(const uint8_t* data, size_t size_bytes) {
    if (!g_ring_buffer || !data || size_bytes < AUDIO_FRAME_BYTES) return 0;
    size_t frames = size_bytes / AUDIO_FRAME_BYTES;
    size_t pushed = 0;
    for (size_t i = 0; i < frames; ++i) {
        const int16_t* frame_ptr = reinterpret_cast<const int16_t*>(data + (i * AUDIO_FRAME_BYTES));
        if (g_ring_buffer->push_frame(frame_ptr)) {
            pushed++;
        } else {
            break;
        }
    }
    return static_cast<int>(pushed);
}

/**
 * Pops 1 frame (320 int16 samples) from the global ring buffer.
 * Returns 1 on success, 0 on buffer empty or error.
 */
extern "C" SIMD_API int ring_buffer_pop_frame(int16_t* out_samples) {
    if (!g_ring_buffer) return 0;
    return g_ring_buffer->pop_frame(out_samples) ? 1 : 0;
}

/**
 * Pops up to num_samples (in 320-sample frame chunks) into out_samples.
 * Returns number of complete frames successfully popped.
 */
extern "C" SIMD_API int ring_buffer_pop(int16_t* out_samples, size_t num_samples) {
    if (!g_ring_buffer || !out_samples || num_samples < AUDIO_FRAME_SAMPLES) return 0;
    size_t frames = num_samples / AUDIO_FRAME_SAMPLES;
    size_t popped = 0;
    for (size_t i = 0; i < frames; ++i) {
        if (g_ring_buffer->pop_frame(out_samples + (i * AUDIO_FRAME_SAMPLES))) {
            popped++;
        } else {
            break;
        }
    }
    return static_cast<int>(popped);
}

/**
 * Pops raw PCM bytes from the ring buffer in 640-byte frame chunks.
 * Returns number of complete frames successfully popped.
 */
extern "C" SIMD_API int ring_buffer_pop_bytes(uint8_t* out_data, size_t max_bytes) {
    if (!g_ring_buffer || !out_data || max_bytes < AUDIO_FRAME_BYTES) return 0;
    size_t frames = max_bytes / AUDIO_FRAME_BYTES;
    size_t popped = 0;
    for (size_t i = 0; i < frames; ++i) {
        int16_t* frame_ptr = reinterpret_cast<int16_t*>(out_data + (i * AUDIO_FRAME_BYTES));
        if (g_ring_buffer->pop_frame(frame_ptr)) {
            popped++;
        } else {
            break;
        }
    }
    return static_cast<int>(popped);
}

/**
 * Returns number of audio frames ready to be read.
 */
extern "C" SIMD_API size_t ring_buffer_available_read(void) {
    if (!g_ring_buffer) return 0;
    return g_ring_buffer->available_read();
}

/**
 * Returns remaining frame write capacity before full.
 */
extern "C" SIMD_API size_t ring_buffer_available_write(void) {
    if (!g_ring_buffer) return 0;
    return g_ring_buffer->available_write();
}

/**
 * Clears ring buffer contents.
 */
extern "C" SIMD_API void ring_buffer_clear(void) {
    if (g_ring_buffer) {
        g_ring_buffer->clear();
    }
}

/**
 * Destroys and frees global ring buffer.
 */
extern "C" SIMD_API void ring_buffer_cleanup(void) {
    if (g_ring_buffer) {
        delete g_ring_buffer;
        g_ring_buffer = nullptr;
    }
}

// Instance-based API for multi-channel / multi-buffer architectures
extern "C" SIMD_API void* ring_buffer_create(size_t capacity_frames) {
    auto* rb = new (std::nothrow) SPSCAudioRingBuffer(capacity_frames > 0 ? capacity_frames : DEFAULT_RING_BUFFER_CAPACITY);
    if (!rb || !rb->is_valid()) {
        delete rb;
        return nullptr;
    }
    return static_cast<void*>(rb);
}

extern "C" SIMD_API void ring_buffer_destroy(void* handle) {
    if (handle) {
        delete static_cast<SPSCAudioRingBuffer*>(handle);
    }
}

extern "C" SIMD_API int ring_buffer_handle_push_frame(void* handle, const int16_t* samples) {
    if (!handle) return 0;
    return static_cast<SPSCAudioRingBuffer*>(handle)->push_frame(samples) ? 1 : 0;
}

extern "C" SIMD_API int ring_buffer_handle_pop_frame(void* handle, int16_t* out_samples) {
    if (!handle) return 0;
    return static_cast<SPSCAudioRingBuffer*>(handle)->pop_frame(out_samples) ? 1 : 0;
}

extern "C" SIMD_API size_t ring_buffer_handle_available_read(void* handle) {
    if (!handle) return 0;
    return static_cast<SPSCAudioRingBuffer*>(handle)->available_read();
}

extern "C" SIMD_API size_t ring_buffer_handle_available_write(void* handle) {
    if (!handle) return 0;
    return static_cast<SPSCAudioRingBuffer*>(handle)->available_write();
}
