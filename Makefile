run:
	python3 model.py ${ARGS}
preprocess: 
	unzip Alzheimers.zip
	unzip NonAlzheimers.zip 
	python3 preprocess.py
