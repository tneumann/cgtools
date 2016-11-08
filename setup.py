from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
if use_cython:
    ext = 'pyx'
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext = 'c'

ext_modules = [
    Extension(
        "cgtools.fastmath._fastmath", 
        ["cgtools/fastmath/_fastmath." + ext],
        include_path=[numpy.get_include()],
    ),
]

setup(
  name = 'cgtools',
  version = "0.0.1",
  author = "Thomas Neumann",
  author_email = "neumann.thomas@gmail.com",
  description = "Tools computer graphics and vision, mostly for numpy / scipy",
  license = "MIT",
  packages = ['cgtools'],
  cmdclass = cmdclass,
  ext_modules = ext_modules,
)

