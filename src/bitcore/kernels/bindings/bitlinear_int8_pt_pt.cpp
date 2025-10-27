#include <torch/extension.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <vector>

// Helper function to compute weight scale for per-tensor quantization
torch::Tensor compute_weight_scale_per_tensor(torch::Tensor weight) {
    auto max_abs = torch::abs(weight).max();
    return max_abs / 127.0;
}

// Helper function to compute activation scale for per-tensor quantization
torch::Tensor compute_activation_scale_per_tensor(torch::Tensor x) {
    auto max_abs = torch::abs(x).max();
    return max_abs / 127.0;
}

// Optimized C++ implementation for quantize_weights
std::pair<torch::Tensor, torch::Tensor> bitlinear_int8_pt_pt_quantize_weights(
    torch::Tensor weight
) {
    // Compute weight scale for per-tensor quantization
    auto qweight_scale = compute_weight_scale_per_tensor(weight);
    
    // Quantize weights to int8
    auto qweight = torch::round(weight / qweight_scale).clamp(-127, 127).to(torch::kInt8);
    
    return std::make_pair(qweight_scale, qweight);
}

// Optimized C++ implementation for forward pass
torch::Tensor bitlinear_int8_pt_pt_forward(
    torch::Tensor x,
    torch::Tensor qweight_scale,
    torch::Tensor qweight,
    torch::Tensor bias
) {
    // Compute activation scale for per-tensor quantization
    auto qx_scale = compute_activation_scale_per_tensor(x);
    
    // Quantize activations to int8
    auto qx = torch::round(x / qx_scale).clamp(-127, 127).to(torch::kInt8);
    
    // Dequantize activations back to float32
    auto dx = qx.to(torch::kFloat32) * qx_scale;
    
    // Dequantize weights back to float32
    auto dweight = qweight.to(torch::kFloat32) * qweight_scale;
    
    // Perform linear operation
    torch::Tensor output;
    if (bias.defined()) {
        output = torch::linear(dx, dweight, bias);
    } else {
        output = torch::linear(dx, dweight);
    }
    
    return output;
}

PYBIND11_MODULE(bitlinear_int8_pt_pt_cpp, m) {
    m.doc() = "C++ bindings for BitLinear INT8 per-tensor operations";
    
    m.def("quantize_weights", &bitlinear_int8_pt_pt_quantize_weights, 
          "Prepare weights for INT8 per-tensor quantization",
          py::arg("weight"));
    
    m.def("forward", &bitlinear_int8_pt_pt_forward, 
          "Forward pass for INT8 per-tensor quantized linear operation",
          py::arg("x"), py::arg("qweight_scale"), py::arg("qweight"), py::arg("bias"));
}
