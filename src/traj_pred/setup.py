#!/usr/bin/env python3

import os
import re
import sys
import platform
import subprocess

from packaging.version import Version
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError("CMake must be installed to build the extensions") from e

        cmake_version = Version(
            re.search(r'version\s*([\d.]+)', out.decode()).group(1)
        )
        if platform.system() == "Windows" and cmake_version < '3.1.0':
            raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        active_prefix = os.environ.get("VIRTUAL_ENV", sys.prefix)
        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={active_prefix}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            "-DCGAL_DIR=/usr/local/lib/CGAL-6.0.1", # Remeber to modify this to match your filesystem
        ]

        try:
            pybind11_cmake = subprocess.check_output(
                [sys.executable, "-m", "pybind11", "--cmakedir"]
            ).decode().strip()
            cmake_args.append(f"-Dpybind11_DIR={pybind11_cmake}")
        except subprocess.CalledProcessError:
            # fallback: rely on CMAKE_PREFIX_PATH above
            pass

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f"-DCMAKE_BUILD_TYPE={cfg}"]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = env.get('CXXFLAGS', '') + \
            f' -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name="fsp",
    version="0.1.0",
    description="Scenario optimization based social navigation",
    author="Abbas Shaikh",
    author_email="abbas22shk@gmail.com",
    url="https://github.com/abbashaikh/free-space-social-navigation.git",
    python_requires=">=3.9",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    ext_modules=[
        CMakeExtension(
        'support_pred.halfplane_module',
        sourcedir='src/support_pred'
        )
    ],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
