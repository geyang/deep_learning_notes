author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)

default:
	make install
	make setup-vis-server
	make on-mac
	make train
install:
	pip install -r requirements.txt
setup-vis-server:
	python -m visdom.server > visdom.log 2>&1 &
	sleep 0.5s
on-mac:
	open http://localhost:8097/env/Grammar-Variational-Autoencoder-experiment
train:
	python grammar_vae.py
evaluate:
	python grammar_vae.py --evaluate
