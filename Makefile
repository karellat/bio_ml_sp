run:
	python3 model.py ${ARGS}
preprocess: 
	mkdir -p data
	unzip Alzheimers.zip
	unzip NonAlzheimers.zip 
	mv -n Alzheimers ./data/
	mv -n NonAlzheimers ./data/
