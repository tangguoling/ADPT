# Third-party software notices

ADPT Toolbox uses third-party software. Those components are not licensed under
the ADPT Toolbox Research Edition License; each remains subject to its own
license. The release archive lists runtime requirements but does not bundle the
Python, TensorFlow, CUDA, or cuDNN distributions. Users obtain those packages
from their respective distributors.

The principal direct dependencies are listed below. The package version
installed by a user may have additional dependencies and notices.

| Component | License | Project |
|---|---|---|
| TensorFlow | Apache License 2.0 | https://github.com/tensorflow/tensorflow |
| Keras | Apache License 2.0 | https://github.com/keras-team/keras |
| NumPy | BSD 3-Clause | https://github.com/numpy/numpy |
| SciPy | BSD 3-Clause | https://github.com/scipy/scipy |
| pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas |
| PyTables | BSD 3-Clause | https://github.com/PyTables/PyTables |
| scikit-image | BSD 3-Clause | https://github.com/scikit-image/scikit-image |
| tifffile | BSD 3-Clause | https://github.com/cgohlke/tifffile |
| OpenCV / opencv-python | Apache License 2.0 | https://github.com/opencv/opencv |
| Pillow | HPND License | https://github.com/python-pillow/Pillow |
| Matplotlib | PSF-based license | https://github.com/matplotlib/matplotlib |
| PyYAML | MIT License | https://github.com/yaml/pyyaml |
| imgaug | MIT License | https://github.com/aleju/imgaug |
| tqdm | MPL-2.0 and MIT | https://github.com/tqdm/tqdm |
| pydot | MIT License | https://github.com/pydot/pydot |
| Intel RealSense / pyrealsense2 | Apache License 2.0 | https://github.com/IntelRealSense/librealsense |
| Cython | Apache License 2.0 | https://github.com/cython/cython |

Optional NVIDIA CUDA, cuDNN, cuBLAS, CUPTI, and NVCC packages are supplied by
NVIDIA and are governed by NVIDIA's applicable license terms. They are not
redistributed in the ADPT release archive.

The authoritative license text and copyright notices for every installed
package are included in that package's distribution metadata. If an ADPT
release later bundles any third-party binary, its complete required notices and
license text must be added to the archive before distribution.
