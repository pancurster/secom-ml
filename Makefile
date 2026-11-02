.PHONY: lint
.PHONY: test

lint:
	pylint src/secomml

test:
	pytest tests/
