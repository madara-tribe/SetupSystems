FROM nvidia/cuda:10.1-devel
ENV DEBIAN_FRONTEND=noninteractive

# Install basics
WORKDIR /home/app
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends \
    python3-dev \
    python3-wheel \
    python3-setuptools \
    g++

RUN apt install -y \
    sudo tzdata vim git cmake wget unzip build-essential libbz2-dev tk-dev

RUN apt-get install -y python3 python3-pip ffmpeg
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


# Install python library
RUN pip3 install --upgrade pip
RUN pip3 install 'chainer<6.0.0'
RUN pip3 install cupy-cuda101==7.8.0 
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install world4py dill 
CMD [ "/bin/bash" ]
