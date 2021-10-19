cd build
cmake --build . --parallel 12 --config Debug
gdb -ex run ./kandi3dloc
cd ..

