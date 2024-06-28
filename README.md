# Accelerated VPint
This repository contains the GPU accelerated version and the multiprocessing version of the original VPint algorithm by Arp et al. (2022). \
\
The original VPint algorithm can be found at: https://github.com/ADA-research/VPint \
The paper on the original VPint algorithm can be found at: https://doi.org/10.1007/s10618-022-00843-2 \
\
\
\
The "run" files in the AcceleratedVersions directory contain the code for the multiprocessing method and the GPU accelerated method in the corresponding files. These files are necessary to run the methods.

The MRP, WP_MRP and SD_MRP files in the AcceleratedVersions/VPint directory only contain code from the original VPint algorithm by Arp et al. (2022). The multiprocessing method uses these files.\
\
The cupy_MRP and cupy_WP_MRP files in AcceleratedVersions/VPint contain the adapted version for the GPU accelerated version, with CuPy instead of NumPy for many functions and datatypes.\
