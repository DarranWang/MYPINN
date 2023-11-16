#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> feconvR_cuda_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor filters);

std::vector<torch::Tensor> feconvR_cuda_backward(
	torch::Tensor gradV,
	torch::Tensor H8types,
	torch::Tensor filters);

/*
// NOTE: torch_ASSERT has become torch_CHECK on master after 0.4.
#define CHECK_CUDA(x) torch_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) torch_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
*/
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> feconvR_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor filters){
	
	CHECK_INPUT(U);
	CHECK_INPUT(H8types);
	CHECK_INPUT(filters);
	
	//torch::DeviceGuard guard(U.device());

	return feconvR_cuda_forward(U,H8types,filters);
}

std::vector<torch::Tensor> feconvR_backward(
	torch::Tensor gradV,
	torch::Tensor H8types,
	torch::Tensor filters){

	CHECK_INPUT(gradV);
	// CHECK_INPUT(H8types);
	// CHECK_INPUT(filters);
	
	return feconvR_cuda_backward(gradV,H8types,filters);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &feconvR_forward, "FECONVR forward (CUDA)");
  m.def("backward", &feconvR_backward, "FECONVR backward (CUDA)");
}
