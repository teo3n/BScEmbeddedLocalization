# kandi3d

```
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
cd ..
ln build/compile_commands.json .
```

## how to 
Sender: Comment out  ```#define IS_SERVER``` and ```#define USE_OPEN3D``` in src/constants.h. </br>

Receiver: Make sure those are not commented out. </br> </br>

Make sure the ports and IPs are correct, run the receiver, and then run the sender. Wait until no more data is sent, the receiver will visualize the points and cameras.
