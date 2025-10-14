
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="decompress_render",
    packages=["decompress_render"],
    ext_modules=[
        CUDAExtension(
            name="decompress_render._C",
            sources=[
                "src/utils.cu",
                "binding.cpp"
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)