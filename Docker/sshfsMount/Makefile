build:
	docker image build . -t ssh_server
run:
	docker run -d -p 10000:22 ssh_server
# passward =="screencast"
login:
	ssh root@127.0.0.1 -p 10000
