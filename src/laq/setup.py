from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='laq_cuda',
    ext_modules=[
        CUDAExtension(
            name='laq_cuda',
            sources=['quant.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
