.PHONY: env-hf env-cpu lab

env-gpu:
	conda env create -f environment-gpu.yml

env-cpu:
	conda env create -f environment-cpu.yml

lab:
	jupyter lab