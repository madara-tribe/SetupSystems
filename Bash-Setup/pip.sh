#!/bin/sh
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
curl -kL https://bootstrap.pypa.io/get-pip.py | python
pip -V
