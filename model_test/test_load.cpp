#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[]) {
  if (argc != 4) {
    std::cerr << "usage: example-app <path-to-exported-script-module> <h> <w>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "ok\n";

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, atoi(argv[2]), atoi(argv[3])}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}