#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>

#if !defined(DISABLE_OPENMP) && defined(_OPENMP)
    #include <omp.h>
    #define HAS_OPENMP
#endif

// 2-bit encoding helpers for {-1,0,+1}
static inline uint8_t enc2(int v) { return v==0 ? 0u : (v>0 ? 1u : 2u); }
static inline int8_t  dec2(uint8_t c){ return c==1 ? +1 : (c==2 ? -1 : 0); }

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

    // Choose an O tile that keeps W_tile (~O_TILE*I floats) in LLC.
    // 512 is a reasonable default; tune per machine.
    const int64_t O_TILE = 512;

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
        // Parallelize over rows; inner loop is tight and cache-friendly.
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int64_t r = 0; r < O_t; ++r) {
            const uint8_t* __restrict prow = &pw[o0 + r][0];
            float* __restrict dst = &wbuf[r][0];

            int64_t i = 0;
            for (int64_t c = 0; c < I4; ++c) {
                const uint8_t byte = prow[c];

                // Extract four 2-bit codes
                // code0: bits [1:0], code1: [3:2], code2: [5:4], code3: [7:6]
                uint8_t code0 = (byte >> 0) & 0x3u;
                uint8_t code1 = (byte >> 2) & 0x3u;
                uint8_t code2 = (byte >> 4) & 0x3u;
                uint8_t code3 = (byte >> 6) & 0x3u;

                // Map codes -> {-1,0,+1}
                // enc: 0->0, 1->+1, 2->-1, 3->0
                if (i < I) { dst[i++] = (code0 == 1 ? +1.f : (code0 == 2 ? -1.f : 0.f)); }
                if (i < I) { dst[i++] = (code1 == 1 ? +1.f : (code1 == 2 ? -1.f : 0.f)); }
                if (i < I) { dst[i++] = (code2 == 1 ? +1.f : (code2 == 2 ? -1.f : 0.f)); }
                if (i < I) { dst[i++] = (code3 == 1 ? +1.f : (code3 == 2 ? -1.f : 0.f)); }
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