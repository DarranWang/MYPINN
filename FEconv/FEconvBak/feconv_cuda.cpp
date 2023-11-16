#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> feconv_cuda_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor nodIdx,
	torch::Tensor filters);

std::vector<torch::Tensor> feconv_cuda_backward(
	torch::Tensor gradU);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> feconv_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor nodIdx,
	torch::Tensor filters){
	
	CHECK_INPUT(U);
	CHECK_INPUT(H8types);
	CHECK_INPUT(nodIdx);
	CHECK_INPUT(filters);

	return feconv_cuda_forward(U,H8types,nodIdx,filters);
}

std::vector<torch::Tensor> feconv_backward(
	torch::Tensor gradU){

	CHECK_INPUT(gradU);
	
	return feconv_cuda_backward(gradU);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &feconv_forward, "FECONV forward (CUDA)");
  m.def("backward", &feconv_backward, "FECONV backward (CUDA)");
}
