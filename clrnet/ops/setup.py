import glob
import os

if os.environ.get('CUDA_HOME') is None:
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
import torch

from torch.utils.cpp_extension import CUDA_HOME

from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]


def get_extensions():
    extensions = []
    op_files = glob.glob(os.path.abspath(os.path.dirname(__file__)) + '/csrc/*.c*')
    extension = torch.utils.cpp_extension.CUDAExtension
    ext_name = 'clrnet_nms_impl'

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
    )

    extensions.append(ext_ops)

    return extensions

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "nms_impl",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="nms_impl",
    version="1.0",
    author="clrnet",
    url="https://github.com/Turoad/CLRNet",
    description=
    "PyTorch Wrapper for CUDA Functions of nms",
    packages=find_packages(exclude=(
        "configs",
        "tests",
    )),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
