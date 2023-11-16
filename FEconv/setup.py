from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='feconv_cuda',
    ext_modules=[
        CUDAExtension('feconv_cuda', [
            'feconv_cuda.cpp',
            'feconv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
