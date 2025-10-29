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
    
    const int64_t B = input.size(0);
    const int64_t I = input.size(1);
    const int64_t O = packed_weights.size(0);
    const int64_t I4 = (I + 3) / 4;
    
    TORCH_CHECK(packed_weights.size(1) == I4, "packed column mismatch: expected ceil(I/4)");
    
    // Unpack weights once to float32 [O, I]
    auto unpacked_weights = torch::empty({O, I}, input.options());
    
    auto p = packed_weights.accessor<uint8_t, 2>();
    auto w = unpacked_weights.accessor<float, 2>();
    
    // Unpack all weights
    for (int64_t o = 0; o < O; ++o) {
        int64_t i_idx = 0;
        for (int64_t c = 0; c < I4; ++c) {
            const uint8_t byte = p[o][c];
            
            // Unpack 4 weights from this byte
            for (int off = 0; off < 4 && i_idx < I; ++off, ++i_idx) {
                const uint8_t code = (byte >> (2*off)) & 0x3u;
                w[o][i_idx] = static_cast<float>(dec2(code));
            }
        }
    }
    
    // Use ATen's optimized matrix multiplication: out = input @ unpacked_weights.T
    // This calls into highly optimized BLAS implementations (MKL, OpenBLAS, Accelerate)
    auto out = at::matmul(input, unpacked_weights.t());
    
    // Apply scale
    out.mul_(static_cast<float>(scale));
    
    // Add bias if provided
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