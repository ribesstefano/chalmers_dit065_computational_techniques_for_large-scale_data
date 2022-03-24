build:
	docker build -t dit-065:ass1 .

shell: build
	docker run -ti --rm -v $(PWD)/src:/usr/src/app dit-065:ass1 bash
