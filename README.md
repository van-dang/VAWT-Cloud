The following instructions were tested with Ubuntu 18.04
# Requirements:
* Singularity container
```bash
sudo apt-get singulariy-container
```

# Build the FEniCS-HPC image
```bash
wget https://raw.githubusercontent.com/van-dang/VAWT-Cloud/master/build_image_source_vawt
sudo singularity build -w writable_fenics_hpc.simg build_fenics_hpc_image_recipe
```
# Test if mpi works correctly with the FEniCS-HPC image
```bash
wget https://raw.githubusercontent.com/wesleykendall/mpitutorial/gh-pages/tutorials/mpi-hello-world/code/mpi_hello_world.c
singularity exec -B $PWD writable_fenics_hpc.simg mpicc mpi_hello_world.c -o mpi_hello_world
singularity exec -B $PWD writable_fenics_hpc.simg mpirun -n 3 ./mpi_hello_world
```
The results would be
```bash
Hello world from processor dmri, rank 0 out of 3 processors
Hello world from processor dmri, rank 1 out of 3 processors
Hello world from processor dmri, rank 2 out of 3 processors
```
# Note
For a multi-node system, openmpi needs to be compatible between the hosted machine and the image to launch with many processors beyond one node. It requires to install the same version of openmpi on the hosted machine and the image. The command to launch the demo is
```bash
mpirun -n 30 singularity exec -B $PWD writable_fenics_hpc.simg ./mpi_hello_world
```

# Simulation with the Marsta turbine

### Pre-processing
mpirun -n 8 -quiet pre-prossessing-vawt/vawt_prepro -m  meshes_vawt/marsta_mesh.xml -o mesh.bin -n 0 -a adaptive -s box -k 0.95 0.005 -c 3.0 0 -44.0 -L 2.8 1. 3.

### Execute the demo
mpirun -n 8 source_vawt/demo -m mesh.bin -n 500 -v 1.45e-5 -a 0 -t 60.83 -p 1 -T 500 -w 200 -b 0.5 -r 49.0 -c 0.1 -k 0.2 10.0 10.0 -d 0 -g 0.1 -x 0.3 -z -50.0 50.0

# References
[1] V. D. Nguyen, J. Jansson, M. Leoni, B. Janssen, A. Goude, and J. Hoffman, “Modelling of rotating vertical axis turbines using a multiphase finite element method,” in MARINE 2017 : Computational Methods in Marine Engineering VII15 - 17 May 2017, Nantes, France, 2017, pp. 950–960, qC 20170629. [Online]. Available: http://congress.cimne.com/marine2017/frontal/Doc/Ebookmarine.pdf

[2] V.-D. Nguyen, J. Jansson, A. Goude, and J. Hoffman, “Direct finite element simulation of the turbulent flow past a vertical axis wind turbine,” Renewable Energy, vol. 135, pp. 238 – 247, 2019. [Online]. Available: http://www.sciencedirect.com/science/article/pii/

[3] V.-D. Nguyen, J. Jansson, A. Goude, and J. Hoffman, “Technical report – comparison of direct finite element simulation with actuator line models and vortex models for simulation of turbulent flow past a vertical axis wind turbine,” 2019. [Online]. Available: https://arxiv.org/abs/1909.01776
