# https://qiita.com/Salinger/items/c7b87d7000e48be6ebe2
do1:
	ipython
	# from notebook.auth import passwd
	# passwd() ==> OUTPUT_ID
do2:
	sudo vi ~/.jupyter/jupyter_notebook_config.py
	c.NotebookApp.ip = '*'
        c.NotebookApp.open_browser = False
        c.NotebookApp.password = ${OUTPUT_ID}

run:
	jupyter notebook &
AD = 54.199.223.28
in:
	http://${AD}:8888/
