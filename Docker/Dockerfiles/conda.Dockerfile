# https://www.eureka-moments-blog.com/entry/2020/02/22/160931
# pull ubuntu 18.04 as base image
FROM ubuntu:18.04

# update packages
RUN set -x && \
    apt update && \
    apt upgrade -y

# install command
RUN set -x && \
    apt install -y wget && \
    apt install -y sudo

# anaconda
RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
    bash Anaconda3-2020.02-Linux-x86_64.sh && \
    rm Anaconda3-2020.02-Linux-x86_64.sh

# path setteing
ENV PATH $PATH:/root/anaconda3/bin

# python library
WORKDIR /root
ADD requirements.txt /root
RUN pip install -r requirements.txt

# move to root directory
WORKDIR ../
