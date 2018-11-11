cd train/darknet/
make -j20  &> buid.log && echo "cpu build success!!!" || echo "cpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_cpu.so
cp Makefile.gpu Makefile
make -j20 &> build.log && echo "gpu build success!!!" || echo "gpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_gpu.so
cp Makefile.cpu Makefile
cd -
