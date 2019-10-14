cd ..
make clean
make all
cd test
mpirun -np 2 ./gf2_mpi
