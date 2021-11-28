FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
RUN DEBIAN_FRONTEND=noninteractive


# update packages
RUN set -x && \
    apt update && apt upgrade -y

# install command
RUN set -x && apt install -y sudo vim git cmake wget unzip build-essential libbz2-dev tk-dev


RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b && \
    rm Anaconda3-2021.05-Linux-x86_64.sh

# path setteing
ENV PATH $PATH:/root/anaconda3/bin

WORKDIR /root
RUN conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.1 -c pytorch
RUN conda install -c conda-forge librosa=0.6.1
RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN apt install -y ffmpeg sox
RUN apt-get install -y yasm libsndfile1-dev
RUN rm /tmp/*

WORKDIR /opt
CMD ["/bin/bash"]
