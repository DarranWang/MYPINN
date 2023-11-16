#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void feconvR_cuda_forward_kernel(
	const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> U,
	const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> H8types,
	const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> filters,
	torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> KU)
{	
	const int outidx = blockIdx.x / 41; // 0 - 17
	//const int Idxx = threadIdx.x % 41;
	//const int Idxy = threadIdx.x / 41;
	const int Idxx = blockIdx.x % 41; // 0 - 40
	const int Idxy = blockIdx.y;
	const int Idxz = blockIdx.z;
	
	const int h8type = H8types[threadIdx.x][0][Idxx][Idxy][Idxz];
	// const auto fkernels = filters[h8type];

	// scalar_t convresult = 0.0;

	// int direction = outidx % 3;
	// for (int j = 0; j < 27; j++)
	// 	{
	// 		int uidx1 = nodIdx[Idxx][Idxy][Idxz][j][0];
	// 		int uidx2 = nodIdx[Idxx][Idxy][Idxz][j][1];
	// 		int uidx3 = nodIdx[Idxx][Idxy][Idxz][j][2];
	// 		if ((uidx1+1)*(uidx2+1)*(uidx3+1)!=0)
	// 		{
	// 			for (int ix= 0; ix < 3; ix++)
	// 			{
	// 				convresult += U[threadIdx.x][outidx - direction + ix][uidx1][uidx2][uidx3] * fkernels[direction][ix][j];

	// 			}
	// 		}
	// 	}
	// KU[threadIdx.x][outidx][Idxx][Idxy][Idxz] = convresult;

	KU[threadIdx.x][outidx][Idxx][Idxy][Idxz] = U[threadIdx.x][outidx][Idxx][Idxy][Idxz] * filters[h8type][outidx];
}

template <typename scalar_t>
__global__ void feconvR_cuda_backward_kernel(
	const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> gradV,
	const torch::PackedTensorAccessor32<int,5,torch::RestrictPtrTraits> H8types,
	const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> filters,
	torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> d_U)
{	
	const int outidx = blockIdx.x / 41; // 0 - 17
	const int Idxx = blockIdx.x % 41; // 0 - 40
	const int Idxy = blockIdx.y;
	const int Idxz = blockIdx.z;

	const int h8type = H8types[threadIdx.x][0][Idxx][Idxy][Idxz];
	// const auto fkernels = filters[h8type];

	// scalar_t convresult = 0.0;

	// int direction = outidx % 3;
	// for (int j = 0; j < 27; j++)
	// 	{
	// 		int uidx1 = nodIdx[Idxx][Idxy][Idxz][j][0];
	// 		int uidx2 = nodIdx[Idxx][Idxy][Idxz][j][1];
	// 		int uidx3 = nodIdx[Idxx][Idxy][Idxz][j][2];
	// 		if ((uidx1+1)*(uidx2+1)*(uidx3+1)!=0)
	// 		{
	// 			for (int ix= 0; ix < 3; ix++)
	// 			{
	// 				convresult += gradV[threadIdx.x][outidx - direction + ix][uidx1][uidx2][uidx3] * fkernels[direction][ix][j];

	// 			}
	// 		}
	// 	}
	// d_U[threadIdx.x][outidx][Idxx][Idxy][Idxz] = convresult;
	d_U[threadIdx.x][outidx][Idxx][Idxy][Idxz] = gradV[threadIdx.x][outidx][Idxx][Idxy][Idxz] * filters[h8type][outidx];
}




std::vector<torch::Tensor> 
feconvR_cuda_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor filters)
{

	const auto batch_size = U.size(0);

	auto KU = torch::zeros_like(U);

	// const dim3 blocks(41,41,41);
	// const dim3 threads(18,batch_size);
	const dim3 blocks(18*41,41,41);
	const dim3 threads(batch_size);

	//const dim3 blocks(18,batch_size);
	//const dim3 threads(41,41,41);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(U.scalar_type(), "feconvR_forward_cuda", ([&] {
    feconvR_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        U.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        H8types.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        filters.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        KU.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

  return {KU};

}

std::vector<torch::Tensor> 
feconvR_cuda_backward(
	torch::Tensor gradV,
	torch::Tensor H8types,
	torch::Tensor filters)
{
	const auto batch_size = gradV.size(0);
	auto d_U = torch::zeros_like(gradV);

	const dim3 blocks(18*41,41,41);
	const dim3 threads(batch_size);

	AT_DISPATCH_FLOATING_TYPES_AND_HALF(gradV.scalar_type(), "feconvR_backward_cuda", ([&] {
    feconvR_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
    	gradV.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
        H8types.packed_accessor32<int,5,torch::RestrictPtrTraits>(),
        filters.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_U.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>());
  }));

	return {d_U};
}
