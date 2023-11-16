#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void feconv_cuda_forward_kernel2(
	const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> U,
	const torch::PackedTensorAccessor<int,5,torch::RestrictPtrTraits,size_t> H8types,
	const torch::PackedTensorAccessor<int,5,torch::RestrictPtrTraits,size_t> nodIdx,
	const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> filters,
	torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> KU)
{	
	const int outidx = blockIdx.x / 41; // 0 - 17
	const int Idxx = blockIdx.x % 41; // 0 - 40
	//const int Idxx = threadIdx.x % 41;
	//const int Idxy = threadIdx.x / 41;
	// const int Idxx = blockIdx.x;
	const int Idxy = blockIdx.y;
	const int Idxz = blockIdx.z;

	const int h8type = H8types[threadIdx.x][0][Idxx][Idxy][Idxz];
	//const auto fkernels = filters[h8type];

	scalar_t convresult = 0.0;

	int direction = outidx % 3;

	for (int ix= 0; ix < 3; ix++)
	{
		for (int j = 0; j < 27; j++)
		{
			int uidx1 = nodIdx[Idxx][Idxy][Idxz][j][0];
			int uidx2 = nodIdx[Idxx][Idxy][Idxz][j][1];
			int uidx3 = nodIdx[Idxx][Idxy][Idxz][j][2];
			if ((uidx1+1)*(uidx2+1)*(uidx3+1)!=0)
			{
				convresult += U[threadIdx.x][outidx - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][3 * direction + ix][j];
			}
		}
	}
	KU[threadIdx.x][outidx][Idxx][Idxy][Idxz] = convresult;
}


template <typename scalar_t>
__global__ void feconv_cuda_forward_kernel(
	const torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> U,
	const torch::PackedTensorAccessor<int,5,torch::RestrictPtrTraits,size_t> H8types,
	const torch::PackedTensorAccessor<int,5,torch::RestrictPtrTraits,size_t> nodIdx,
	const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> filters,
	torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> KU)
{	
	//const int Idxx = threadIdx.x % 41;
	//const int Idxy = threadIdx.x / 41;
	const int Idxx = blockIdx.x;
	const int Idxy = blockIdx.y;
	const int Idxz = blockIdx.z;
	const int h8type = H8types[threadIdx.y][0][Idxx][Idxy][Idxz];
	const auto fkernels = filters[h8type];

	scalar_t convresult = 0.0;

	int direction = threadIdx.x % 3;
/**/
	for (int ix= 0; ix < 3; ix++)
	{
		for (int j = 0; j < 27; j++)
		{
			int uidx1 = nodIdx[Idxx][Idxy][Idxz][j][0];
			int uidx2 = nodIdx[Idxx][Idxy][Idxz][j][1];
			int uidx3 = nodIdx[Idxx][Idxy][Idxz][j][2];
			if ((uidx1+1)*(uidx2+1)*(uidx3+1)!=0)
			{
				convresult += U[threadIdx.y][threadIdx.x - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][3 * direction + ix][j];
//				convresult += U[threadIdx.y][threadIdx.x - direction + ix][uidx1][uidx2][uidx3] * filters[255][3 * direction + ix][j];
//				convresult += filters[h8type][3 * direction + ix][j];
			}
		}
	}
	//KU[blockIdx.x][blockIdx.y][threadIdx.x][threadIdx.y][threadIdx.z] = 1.0;
	KU[threadIdx.y][threadIdx.x][Idxx][Idxy][Idxz] = convresult;
	/*
	const int h8type = H8types[blockIdx.y][blockIdx.x][threadIdx.x][threadIdx.y][threadIdx.z];
	const auto fkernels = filters[h8type];

	scalar_t convresult = 0.0;

	//blockIdx.y % 3 == 0: filters[0:3]
	//blockIdx.y % 3 == 1: filters[3:6]
	//blockIdx.y % 3 == 2: filters[6:9]
	int direction = blockIdx.x % 3;

	for (int ix= 0; ix < 3; ix++)
	{
		for (int j = 0; j < 27; j++)
		{
			int uidx1 = nodIdx[threadIdx.x][threadIdx.y][threadIdx.z][j][0];
			int uidx2 = nodIdx[threadIdx.x][threadIdx.y][threadIdx.z][j][1];
			int uidx3 = nodIdx[threadIdx.x][threadIdx.y][threadIdx.z][j][2];
			convresult += 1.0;//U[blockIdx.x][blockIdx.y - direction + ix][uidx1][uidx2][uidx3] * filters[h8type][3 * direction + ix][j];
		}
	}
	// scalar_t convresult = h8type*1.0;
	//KU[blockIdx.x][blockIdx.y][threadIdx.x][threadIdx.y][threadIdx.z] = 1.0;
	KU[blockIdx.y][blockIdx.x][threadIdx.x][threadIdx.y][threadIdx.z] = convresult;*/
}



std::vector<torch::Tensor> 
//torch::Tensor 
feconv_cuda_forward(
	torch::Tensor U,
	torch::Tensor H8types,
	torch::Tensor nodIdx,
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
	// const dim3 blocks(batch_size,3);
	// const dim3 threads(11,11,11);
	/**/
	AT_DISPATCH_FLOATING_TYPES(U.type(), "feconv_forward_cuda", ([&] {
    feconv_cuda_forward_kernel2<scalar_t><<<blocks, threads>>>(
        U.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>(),
        H8types.packed_accessor<int,5,torch::RestrictPtrTraits,size_t>(),
        nodIdx.packed_accessor<int,5,torch::RestrictPtrTraits,size_t>(),
        filters.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
        KU.packed_accessor<scalar_t,5,torch::RestrictPtrTraits,size_t>());
  }));

  return {KU};
  //return {U};
}

std::vector<torch::Tensor> 
// torch::Tensor 
feconv_cuda_backward(
	torch::Tensor gradU)
{
	auto dU = torch::zeros_like(gradU);
	return {dU};
}
