rm -rf build
mkdir build
cd build
cmake ..
cmake --build . --parallel 12 --config Debug
./kandi3dloc
cd ..