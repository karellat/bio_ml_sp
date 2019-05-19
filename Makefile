run:
	python3 runner.py ${ARGS}
preprocess: 
	unzip Alzheimers.zip
	unzip NonAlzheimers.zip 
	python3 preprocess.py
