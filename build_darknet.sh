cd train/darknet/
make -j20  &> cpu_build.log && echo "cpu build success!!!" || echo "cpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_cpu.so
cp Makefile.gpu Makefile
make -j20 &> gpu_build.log && echo "gpu build success!!!" || echo "gpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_gpu.so
cp Makefile.cpu Makefile
cd -
