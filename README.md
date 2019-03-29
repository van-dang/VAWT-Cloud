# VAWT-Cloud

## For Marsta turbine

### Pre-processing
mpirun -n 8 -quiet pre-prossessing-vawt/vawt_prepro -m  meshes_vawt/marsta_mesh.xml -o mesh.bin -n 0 -a adaptive -s box -k 0.95 0.005 -c 3.0 0 -44.0 -L 2.8 1. 3.

### Execute the demo
mpirun -n 8 source_vawt/demo -m mesh.bin -n 500 -v 1.45e-5 -a 0 -t 60.83 -p 1 -T 500 -w 200 -b 0.5 -r 49.0 -c 0.1 -k 0.2 10.0 10.0 -d 0 -g 0.1 -x 0.3 -z -50.0 50.0
