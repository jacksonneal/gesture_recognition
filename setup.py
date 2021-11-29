from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("models/*.pyx"))

# setup(
#     name='gesture_recognition',
#     version='0.1.0',
#     packages=find_packages(include=['gesture_recognition', 'gesture_recognition.*'])
# )
