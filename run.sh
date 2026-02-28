mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j32
cd ..
./build/test_runner