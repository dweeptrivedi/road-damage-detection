cd train/darknet/
make -j20
cp libdarknet.so ../../src/libdarknet_cpu.so
cp Makefile.gpu Makefile
make -j20
cp libdarknet.so ../../src/libdarknet_gpu.so
cp Makefile.cpu Makefile
cd -
