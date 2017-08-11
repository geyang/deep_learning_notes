author=$(Ge Yang)
author_email=$(yangge1987@gmail.com)
# notes on python packaging: http://python-packaging.readthedocs.io/en/latest/minimal.html
default: ;
wheel:
	rm -rf dist
	python setup.py bdist_wheel
dev:
	make wheel
	pip install dist/waterbear*.whl
publish:
	make test
	make wheel
	twine upload dist/*
test:
	pytest
