from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "falling_trees.gravitree_cpp",
            ["falling_trees/gravitree.cpp"],
            cxx_std=17,
            define_macros=[("FALLING_TREES_PYBIND", None)],
            extra_compile_args=["-O3"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
