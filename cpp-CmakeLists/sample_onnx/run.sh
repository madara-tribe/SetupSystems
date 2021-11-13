# !/bin/sh
cmake -B build
cmake --build build --config Release --parallel
cd build/src/
#./inference  --use_cpu
./inference  --use_cuda

