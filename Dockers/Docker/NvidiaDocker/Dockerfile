FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHON_VERSION 3.7.0
ARG GPU_ARCH='7.5'
WORKDIR /home/app


# Build tools
RUN apt update && apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential
RUN apt-get -y install ffmpeg

# Media I/O:
RUN sudo apt install -y \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libgtk2.0-dev

# Video I/O:
RUN sudo apt install -y \
    libdc1394-22-dev libavcodec-dev libavformat-dev \
    libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev \
    libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Parallelism and linear algebra libraries:
RUN sudo apt install -y libtbb-dev libeigen3-dev

# python 3.7 and pip3
RUN apt-get update && apt-get upgrade -y && \
    apt install -y  --no-install-recommends make cmake gcc git g++ unzip wget build-essential zlib1g-dev libffi-dev libssl-dev && \
    apt clean && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar zxf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure && \
    make altinstall



RUN ln -s /usr/local/bin/python3.7 /bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /bin/pip3

# pytorch
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python tqdm joblib


# ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
