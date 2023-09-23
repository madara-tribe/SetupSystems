FROM python:3


WORKDIR /home/app
# pwd => /home/app

ADD venv /home/app/

# pytorch
RUN pip install --upgrade pip
RUN pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy tqdm torchvision opencv-python

# ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
