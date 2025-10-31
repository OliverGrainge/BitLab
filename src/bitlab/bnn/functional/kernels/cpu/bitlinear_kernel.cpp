#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>
#include <algorithm>

#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #include <omp.h>
    #define HAS_OPENMP
#endif

// SIMD headers for vectorized operations
#ifdef __AVX2__
    #include <immintrin.h>
    #define HAS_AVX2
#endif

#ifdef __SSE4_1__
    #include <smmintrin.h>
    #define HAS_SSE4_1
#endif

// 2-bit encoding helpers for {-1,0,+1}
static inline uint8_t enc2(int v) { return v==0 ? 0u : (v>0 ? 1u : 2u); }
static inline int8_t  dec2(uint8_t c){ return c==1 ? +1 : (c==2 ? -1 : 0); }

// Lookup table for fast code-to-float conversion (eliminates branches)
static const float CODE_TO_FLOAT[4] = {0.0f, +1.0f, -1.0f, 0.0f};

// Vectorized unpacking functions
#ifdef HAS_AVX2
// AVX2 implementation for unpacking 4 bytes (16 weights) at once
static inline void unpack_4bytes_avx2(const uint8_t* src, float* dst, int64_t remaining) {
    // Load 4 bytes
    __m128i bytes = _mm_loadu_si128((__m128i*)src);
    
    // Extract individual bytes and expand to 32-bit
    __m256i expanded = _mm256_cvtepu8_epi32(bytes);
    
    // Create masks for bit extraction
    __m256i mask0 = _mm256_set1_epi32(0x3);      // 0b11
    __m256i mask1 = _mm256_set1_epi32(0xC);      // 0b1100
    __m256i mask2 = _mm256_set1_epi32(0x30);     // 0b110000
    __m256i mask3 = _mm256_set1_epi32(0xC0);     // 0b11000000
    
    // Extract 2-bit codes
    __m256i codes0 = _mm256_and_si256(expanded, mask0);
    __m256i codes1 = _mm256_and_si256(_mm256_srli_epi32(expanded, 2), mask0);
    __m256i codes2 = _mm256_and_si256(_mm256_srli_epi32(expanded, 4), mask0);
    __m256i codes3 = _mm256_and_si256(_mm256_srli_epi32(expanded, 6), mask0);
    
    // Convert to float using lookup table
    __m256 floats0 = _mm256_i32gather_ps(CODE_TO_FLOAT, codes0, 4);
    __m256 floats1 = _mm256_i32gather_ps(CODE_TO_FLOAT, codes1, 4);
    __m256 floats2 = _mm256_i32gather_ps(CODE_TO_FLOAT, codes2, 4);
    __m256 floats3 = _mm256_i32gather_ps(CODE_TO_FLOAT, codes3, 4);
    
    // Store results (only up to remaining elements)
    int64_t store_count = std::min(static_cast<int64_t>(16), remaining);
    for (int64_t i = 0; i < store_count; ++i) {
        dst[i] = ((float*)&floats0)[i];
    }
    if (store_count > 4) {
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(4), store_count - 4); ++i) {
            dst[4 + i] = ((float*)&floats1)[i];
        }
    }
    if (store_count > 8) {
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(4), store_count - 8); ++i) {
            dst[8 + i] = ((float*)&floats2)[i];
        }
    }
    if (store_count > 12) {
        for (int64_t i = 0; i < std::min(static_cast<int64_t>(4), store_count - 12); ++i) {
            dst[12 + i] = ((float*)&floats3)[i];
        }
    }
}
#endif

#ifdef HAS_SSE4_1
// SSE4.1 implementation for unpacking 2 bytes (8 weights) at once
static inline void unpack_2bytes_sse4(const uint8_t* src, float* dst, int64_t remaining) {
    // Load 2 bytes
    __m128i bytes = _mm_loadl_epi64((__m128i*)src);
    
    // Expand to 32-bit
    __m128i expanded = _mm_cvtepu8_epi32(bytes);
    
    // Extract 2-bit codes
    __m128i codes0 = _mm_and_si128(expanded, _mm_set1_epi32(0x3));
    __m128i codes1 = _mm_and_si128(_mm_srli_epi32(expanded, 2), _mm_set1_epi32(0x3));
    __m128i codes2 = _mm_and_si128(_mm_srli_epi32(expanded, 4), _mm_set1_epi32(0x3));
    __m128i codes3 = _mm_and_si128(_mm_srli_epi32(expanded, 6), _mm_set1_epi32(0x3));
    
    // Convert to float using lookup table
    __m128 floats0 = _mm_i32gather_ps(CODE_TO_FLOAT, codes0, 4);
    __m128 floats1 = _mm_i32gather_ps(CODE_TO_FLOAT, codes1, 4);
    __m128 floats2 = _mm_i32gather_ps(CODE_TO_FLOAT, codes2, 4);
    __m128 floats3 = _mm_i32gather_ps(CODE_TO_FLOAT, codes3, 4);
    
    // Store results
    int64_t store_count = std::min(static_cast<int64_t>(8), remaining);
    for (int64_t i = 0; i < store_count; ++i) {
        dst[i] = ((float*)&floats0)[i];
    }
    if (store_count > 4) {
        for (int64_t i = 0; i < store_count - 4; ++i) {
            dst[4 + i] = ((float*)&floats1)[i];
        }
    }
    if (store_count > 8) {
        for (int64_t i = 0; i < store_count - 8; ++i) {
            dst[8 + i] = ((float*)&floats2)[i];
        }
    }
    if (store_count > 12) {
        for (int64_t i = 0; i < store_count - 12; ++i) {
            dst[12 + i] = ((float*)&floats3)[i];
        }
    }
}
#endif

// Fallback scalar implementation
static inline void unpack_bytes_scalar(const uint8_t* src, float* dst, int64_t count) {
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t byte = src[i];
        
        // Extract four 2-bit codes
        uint8_t code0 = (byte >> 0) & 0x3u;
        uint8_t code1 = (byte >> 2) & 0x3u;
        uint8_t code2 = (byte >> 4) & 0x3u;
        uint8_t code3 = (byte >> 6) & 0x3u;
        
        // Use lookup table for fast conversion
        dst[i*4 + 0] = CODE_TO_FLOAT[code0];
        dst[i*4 + 1] = CODE_TO_FLOAT[code1];
        dst[i*4 + 2] = CODE_TO_FLOAT[code2];
        dst[i*4 + 3] = CODE_TO_FLOAT[code3];
    }
}

// ----------------------------------------------------------------------------
// prepare_weights_cpu:
//   Input:  float32 weights [O, I] ~ s * {-1,0,+1}, epsilon threshold
//   Output: (uint8 packed [O, ceil(I/4)], double scale)
// ----------------------------------------------------------------------------
std::tuple<torch::Tensor, double> prepare_weights_cpu(
    torch::Tensor weights,
    double eps
) {
    // Input validation: ensure weights are on CPU, float32, 2D, and contiguous
    TORCH_CHECK(weights.device().is_cpu(), "weights must be on CPU");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(weights.dim() == 2, "weights must be [out_features, in_features]");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

    // Extract dimensions: O = output features, I = input features
    const int64_t O = weights.size(0);
    const int64_t I = weights.size(1);
    const int64_t I4 = (I + 3) / 4;     // Number of packed bytes needed (4 weights per byte)

    // Calculate per-tensor scale using mean absolute value
    double scale = weights.abs().mean().item<double>();
    if (scale == 0.0) scale = 1.0; // Avoid division by zero for all-zero weights

    // Quantize weights to ternary values {-1,0,+1}
    auto w_quantized = (weights / (scale + eps)).round().clamp(-1.0f, 1.0f);

    // Create output tensor for packed weights: [O, I4] uint8 array
    auto packed = torch::empty({O, I4}, weights.options().dtype(torch::kUInt8));
    
    // Get accessors for efficient element access
    auto w = w_quantized.accessor<float, 2>();
    auto p = packed.accessor<uint8_t, 2>();

    // Pack weights row by row
    for (int64_t o = 0; o < O; ++o) {
        for (int64_t c = 0; c < I4; ++c) {
            uint8_t byte = 0;
            
            // Pack 4 weights into this byte (2 bits per weight)
            for (int off = 0; off < 4; ++off) {
                const int64_t i = c*4 + off;
                uint8_t code = 0;
                
                if (i < I) {
                    float v = w[o][i];
                    int q = 0;
                    if (std::fabs(v) > 0.5f) q = (v > 0.f) ? +1 : -1;
                    code = enc2(q);
                }
                
                byte |= static_cast<uint8_t>((code & 0x3u) << (2*off));
            }
            
            p[o][c] = byte;
        }
    }
    
    return std::make_tuple(packed, scale);
}

// ----------------------------------------------------------------------------
// Adaptive tiling function
//   Chooses optimal tile sizes based on problem dimensions and cache size
// ----------------------------------------------------------------------------
std::pair<int64_t, int64_t> choose_tile_sizes(int64_t B, int64_t I, int64_t O) {
    // Estimate cache size (L3 cache is typically 8-32MB on modern CPUs)
    // We'll be conservative and assume 8MB L3 cache
    const int64_t ESTIMATED_L3_CACHE_BYTES = 8 * 1024 * 1024;
    const int64_t FLOAT_SIZE = sizeof(float);
    
    // Calculate memory footprint for different tile sizes
    auto calculate_memory_footprint = [&](int64_t o_tile, int64_t i_tile) -> int64_t {
        // Input tile: B * i_tile * FLOAT_SIZE
        // Weight tile: o_tile * i_tile * FLOAT_SIZE  
        // Output tile: B * o_tile * FLOAT_SIZE
        return (B * i_tile + o_tile * i_tile + B * o_tile) * FLOAT_SIZE;
    };
    
    // Start with reasonable defaults
    int64_t o_tile = std::min(static_cast<int64_t>(512), O);
    int64_t i_tile = std::min(static_cast<int64_t>(256), I);
    
    // Adjust based on problem size and cache constraints
    if (B <= 32) {
        // Small batch: can use larger tiles
        o_tile = std::min(static_cast<int64_t>(1024), O);
        i_tile = std::min(static_cast<int64_t>(512), I);
    } else if (B >= 256) {
        // Large batch: use smaller tiles to fit in cache
        o_tile = std::min(static_cast<int64_t>(256), O);
        i_tile = std::min(static_cast<int64_t>(128), I);
    }
    
    // Ensure we don't exceed cache capacity
    while (calculate_memory_footprint(o_tile, i_tile) > ESTIMATED_L3_CACHE_BYTES / 2 && 
           (o_tile > 64 || i_tile > 64)) {
        if (o_tile > i_tile) {
            o_tile = std::max(static_cast<int64_t>(64), o_tile / 2);
        } else {
            i_tile = std::max(static_cast<int64_t>(64), i_tile / 2);
        }
    }
    
    // Ensure minimum tile sizes for efficiency
    o_tile = std::max(static_cast<int64_t>(64), o_tile);
    i_tile = std::max(static_cast<int64_t>(64), i_tile);
    
    return {o_tile, i_tile};
}

// ----------------------------------------------------------------------------
// bitlinear_cpu_forward:
//   Input:  input [B,I] float32
//           packed_weights [O, ceil(I/4)] uint8
//           scale (double)
//           bias [O] float32 (optional)
//   Output: [B,O] float32
//   
//   Strategy: Unpack weights once to float32, then use ATen's optimized BLAS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// bitlinear_cpu_forward (tiled):
//   Dequantize weights in O-tiles and immediately GEMM each tile.
//   out[:, o0:o1] = scale * (input @ W_tile^T)  (+ bias later)
// ----------------------------------------------------------------------------
torch::Tensor bitlinear_cpu_forward(
    torch::Tensor input,
    torch::Tensor packed_weights,
    double scale,
    torch::optional<torch::Tensor> bias_opt
) {
    TORCH_CHECK(input.device().is_cpu(), "input must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be [B, I]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    TORCH_CHECK(packed_weights.device().is_cpu(), "packed_weights must be on CPU");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");
    TORCH_CHECK(packed_weights.dim() == 2, "packed_weights must be [O, ceil(I/4)]");
    TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");

    const int64_t B  = input.size(0);
    const int64_t I  = input.size(1);
    const int64_t O  = packed_weights.size(0);
    const int64_t I4 = (I + 3) / 4;
    TORCH_CHECK(packed_weights.size(1) == I4, "packed column mismatch: expected ceil(I/4)");

    // Output [B, O]
    auto out = torch::empty({B, O}, input.options());

    // Choose adaptive tile sizes based on problem dimensions and cache size
    auto [O_TILE, I_TILE] = choose_tile_sizes(B, I, O);

    // Accessors
    auto pw = packed_weights.accessor<uint8_t, 2>();

    // Temporary fp32 buffer for one W tile: [O_t, I] (contiguous, row-major)
    // We reuse a single buffer sized for O_TILE; the last tile may be smaller.
    auto wtile = torch::empty({O_TILE, I}, input.options());
    auto wbuf  = wtile.accessor<float, 2>();

    // Process output rows in tiles
    for (int64_t o0 = 0; o0 < O; o0 += O_TILE) {
        const int64_t O_t = std::min<int64_t>(O_TILE, O - o0);

        // Dequantize packed_weights[o0:o0+O_t, :] -> wtile[0:O_t, :]
        // Parallelize over rows; inner loop uses vectorized unpacking.
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t r = 0; r < O_t; ++r) {
            const uint8_t* __restrict prow = &pw[o0 + r][0];
            float* __restrict dst = &wbuf[r][0];
            
            // Prefetch next row for better cache utilization
            if (r + 1 < O_t) {
                __builtin_prefetch(&pw[o0 + r + 1][0], 0, 3); // Read, high temporal locality
            }

            int64_t i = 0;
            int64_t c = 0;
            
            // Vectorized unpacking for as many bytes as possible
            #ifdef HAS_AVX2
            // Process 4 bytes (16 weights) at a time with AVX2
            while (c + 4 <= I4 && i + 16 <= I) {
                unpack_4bytes_avx2(&prow[c], &dst[i], I - i);
                c += 4;
                i += 16;
            }
            #elif defined(HAS_SSE4_1)
            // Process 2 bytes (8 weights) at a time with SSE4.1
            while (c + 2 <= I4 && i + 8 <= I) {
                unpack_2bytes_sse4(&prow[c], &dst[i], I - i);
                c += 2;
                i += 8;
            }
            #endif
            
            // Fallback to scalar for remaining bytes
            while (c < I4 && i < I) {
                const uint8_t byte = prow[c];
                
                // Extract four 2-bit codes
                uint8_t code0 = (byte >> 0) & 0x3u;
                uint8_t code1 = (byte >> 2) & 0x3u;
                uint8_t code2 = (byte >> 4) & 0x3u;
                uint8_t code3 = (byte >> 6) & 0x3u;

                // Use lookup table for fast conversion
                if (i < I) { dst[i++] = CODE_TO_FLOAT[code0]; }
                if (i < I) { dst[i++] = CODE_TO_FLOAT[code1]; }
                if (i < I) { dst[i++] = CODE_TO_FLOAT[code2]; }
                if (i < I) { dst[i++] = CODE_TO_FLOAT[code3]; }
                
                c++;
            }
        }

        // Compute out[:, o0:o0+O_t] = scale * (input @ (wtile[0:O_t, :])^T)
        // Use addmm with beta=0 and alpha=scale to fuse scaling.
        auto Wt = wtile.narrow(0, 0, O_t).t().contiguous(); // [I, O_t]
        auto out_block = out.narrow(1, o0, O_t);
        at::addmm_out(/*out=*/out_block,
                      /*self=*/out_block,  // ignored since beta=0
                      /*mat1=*/input,      // [B, I]
                      /*mat2=*/Wt,         // [I, O_t]
                      /*beta=*/0.0f,
                      /*alpha=*/static_cast<float>(scale));
    }

    // Add bias if provided (broadcast over batch)
    if (bias_opt.has_value() && bias_opt->defined()) {
        torch::Tensor bias = *bias_opt;
        TORCH_CHECK(bias.device().is_cpu(), "bias must be on CPU");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == O, "bias must be [O]");
        out.add_(bias);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepare_weights",
          &prepare_weights_cpu,
          "Pack ternary weights into 2-bit format (CPU) and return scale",
          py::arg("weights"),
          py::arg("eps") = 0.0);

    m.def("bitlinear_forward",
          &bitlinear_cpu_forward,
          "BitLinear CPU forward with packed weights and scale (optimized with BLAS)",
          py::arg("input"),
          py::arg("packed_weights"),
          py::arg("scale"),
          py::arg("bias") = py::none());
}