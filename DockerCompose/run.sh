# !/bin/sh
NAME=cuda
pip install docker-compose
docker-compose run --rm --service-ports ${NAME} bash 
