#include <torch/extension.h>
void pipeline(std::vector<at::Tensor>, std::vector<int64_t>, double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("pipeline",&pipeline);}
