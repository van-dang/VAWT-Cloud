The following instructions were tested with Ubuntu 18.04
# Requirements:
* Singularity container
```bash
sudo apt-get singulariy-container
```

# Build the FEniCS-HPC image
```bash
wget https://raw.githubusercontent.com/van-dang/MRI-Cloud/singularity_images/build_fenics_hpc_image_recipe
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
