test:
	PYTHONPATH='src/' python -m pytest -v -W ignore::DeprecationWarning --show-capture=stdout
