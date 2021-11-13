# !/bin.sh
# https://github.com/xmba15/onnx_runtime_cpp
# build onnxruntime from source
sudo bash install_onnx_runtime.sh
make all
make apps
# after make apps
./build/examples/TestImageClassification ./data/squeezenet1.1.onnx ./data/images/dog.jpg

