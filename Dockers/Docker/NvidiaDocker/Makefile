NM = pre
ID =
run:
	docker build -t py3 .
        # docker build -t py3:latest .

de:
	docker rmi ${ID} -f

build:
	docker build ./ -t py3:latest

all_de:
	docker image prune

in:
	docker run -v /Users/hagi/desktop/D/venv/:/home/app/ --rm -it py3 bash
   # docker run -v [ホストdir-path]:[コンテナdir-path] --rm -it [image(container name) name] [commnd]
   # --rm  : docker end時にcontainer auto delete

nvidia_in:
	docker run --runtime=nvidia -v /home/lin/pro:/home/app/pro --rm -ti py3:latest bash

in:
	docker run --runtime=nvidia -v /home/hagi:/home/app/space --rm -it kine_train:latest bash
