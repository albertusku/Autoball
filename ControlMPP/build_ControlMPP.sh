sudo mkdir build && cd build
sudo cmake ..
sudo make -j$(nproc)
cd ..
sudo mkdir -p bin
sudo mv build/ControlMPP bin/
sudo rm -rf build
sudo chmod +x bin/ControlMPP