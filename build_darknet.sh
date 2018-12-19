git clone https://github.com/pjreddie/darknet  ./train/darknet
cd train/darknet/
make -j20  &> buid.log && echo "cpu build success!!!" || echo "cpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_cpu.so
make -j20 GPU=1 CUDNN=1 &> build.log && echo "gpu build success!!!" || echo "gpu build failure!!!"
cp libdarknet.so ../../src/libdarknet_gpu.so
cd -
