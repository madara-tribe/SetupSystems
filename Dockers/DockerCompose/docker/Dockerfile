ARG PYTORCH="1.8.1"
ARG CUDA="11.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y git zsh vim tree sudo unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda clean --all

# install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

ENV PYTHONPATH $PYTHONPATH:/work

WORKDIR /work
