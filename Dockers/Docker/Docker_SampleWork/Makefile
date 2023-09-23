NM = pre
ID = 12
run:
	docker build -t py3 .
de:
	docker rmi ${ID} -f

build:
	docker build ./ -t ${NM}:latest
all_de:
	docker image prune
in:
	docker run -v /Users/hagi/desktop/D/venv/:/home/app/ --rm -it py3 bash
   # docker run -v [ホストdir-path]:[コンテナdir-path] --rm -it [image(container name) name] [commnd]
   # --rm  : docker end時にcontainer auto delete
