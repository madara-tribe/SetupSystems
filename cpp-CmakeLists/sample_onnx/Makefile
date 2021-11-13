cpurun:
	docker build -f Docker/onnxruntime-cpu.Dockerfile --no-cache --tag=onnxruntime-cpu:1.6.0 .

cpuin:
	docker run -it --rm -v $(pwd):/mnt onnxruntime-cpu:1.6.0

cudarun:
	docker build -f Docker/onnxruntime-cuda.Dockerfile --no-cache --tag=onnxruntime-cuda:1.6.0 .


cudain:
	docker run -it --rm --gpus device=0 -v $(pwd):/mnt onnxruntime-cuda:1.6.0

