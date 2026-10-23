#include "summary_program_host.hpp"
void bind_concurrent_probe(pybind11::module_ &);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    bind_concurrent_probe(m);
    bind_summary_program(m);
}
