#!/bin/sh
DOCKER_FOLDER = docker_folder
REMOTE_FOLDER = remote_folder
sshfs root@127.0.0.1:/root/${DOCKER_FOLDER} ./${REMOTE_FOLDER} -p 10000
