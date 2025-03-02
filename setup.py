import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmd = [
            "cmake",
            ext.sourcedir,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir + f"/{ext.name}",
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        # Configure the cmake build
        subprocess.check_call(
            cmd,
            cwd=self.build_temp,
        )

        # Build the extension
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release"], cwd=self.build_temp
        )


setup(
    name="pyenvlp",
    version="1.0.0",
    author="nVerses Capital LLC",
    author_email="user@nverses.com",
    description="Python bindings for the Envelope regression model",
    long_description="",
    ext_modules=[CMakeExtension("pyenvlp")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
)
