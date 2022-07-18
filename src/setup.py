import os
import subprocess

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def filter_filename(filename):
    accepted_extensions = ['.cu', '.cpp']#, '.h', '.hpp', '.cuh']
    for ext in accepted_extensions:
        if filename.endswith(ext):
            return True
    return False

def get_files(dirs):
    files = []
    for dir in dirs:
        for root, dirnames, filenames in os.walk(dir):
            for filename in filenames:
                if filter_filename(filename):
                    files.append(os.path.join(root, filename))
    return files

source_dir = os.getcwd()
include_dirs = ['dense', 'utils']
include_dirs = [os.path.join(source_dir, f) for f in include_dirs]
source_files = get_files(include_dirs)

setup(
    name='gated_attn',
    ext_modules=[
        CUDAExtension(
            'gated_attn', 
            source_files)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=["."])

