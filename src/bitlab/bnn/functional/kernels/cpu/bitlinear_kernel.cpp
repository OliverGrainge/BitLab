#include <torch/extension.h>
#include <vector>

// CPU implementation of bitlinear operation
torch::Tensor bitlinear_cpu_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias
) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weights.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    // Get data pointers
    auto input_data = input.accessor<float, 2>();
    auto weights_data = weights.accessor<float, 2>();
    auto output_data = output.accessor<float, 2>();
    
    // Perform matrix multiplication
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_features; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += input_data[b][i] * weights_data[o][i];
            }
            output_data[b][o] = sum;
        }
    }
    
    // Add bias if provided
    if (bias.defined()) {
        auto bias_data = bias.accessor<float, 1>();
        for (int b = 0; b < batch_size; b++) {
            for (int o = 0; o < out_features; o++) {
                output_data[b][o] += bias_data[o];
            }
        }
    }
    
    return output;
}

// CPU implementation of pack weights
torch::Tensor pack_weights_cpu(
    torch::Tensor weights,
    float eps
) {
    auto packed = torch::zeros_like(weights);
    
    auto weights_data = weights.accessor<float, 2>();
    auto packed_data = packed.accessor<float, 2>();
    
    auto out_features = weights.size(0);
    auto in_features = weights.size(1);
    
    for (int o = 0; o < out_features; o++) {
        for (int i = 0; i < in_features; i++) {
            float val = weights_data[o][i];
            float abs_val = std::abs(val);
            float quantized = (abs_val > eps) ? ((val > 0) ? 1.0f : -1.0f) : 0.0f;
            packed_data[o][i] = quantized;
        }
    }
    
    return packed;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_forward", &bitlinear_cpu_forward, "BitLinear CPU forward");
    m.def("pack_weights", &pack_weights_cpu, "Pack weights CPU");
}