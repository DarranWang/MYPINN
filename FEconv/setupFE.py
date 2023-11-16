from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='feconvR_cuda',
    ext_modules=[
        CUDAExtension('feconvR_cuda', [
            'feconvR_cuda.cpp',
            'feconvR_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
