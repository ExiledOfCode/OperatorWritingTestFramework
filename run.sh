mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j32
./test_runner