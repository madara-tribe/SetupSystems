FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV CUDA_VERSION 10.0
ENV TF_VERSION 1.13.1

ENV PYTHON_VERSION 3.7.0
WORKDIR /home/app
RUN apt-get update && apt-get upgrade -y
RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev
# Media I/O:
RUN sudo apt install -y \
    zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libgtk2.0-dev

# Video I/O:
RUN sudo apt install -y \
    libdc1394-22-dev libavcodec-dev libavformat-dev \
    libswscale-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
    libopencore-amrnb-dev libopencore-amrwb-dev libv4l-dev \
    libxine2-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
RUN sudo apt install -y libtbb-dev libeigen3-dev


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
# install pip

RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==${TF_VERSION}
RUN pip3 install keras==2.2.4
RUN pip3 install scikit-learn matplotlib seaborn tqdm pandas opencv-python
RUN apt-get install -y libgl1-mesa-dev
CMD [ "/bin/bash" ]
