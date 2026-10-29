#include <torch/extension.h>
void run(std::vector<at::Tensor>,std::vector<int64_t>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
