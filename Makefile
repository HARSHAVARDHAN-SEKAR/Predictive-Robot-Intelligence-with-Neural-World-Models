# Neural World Model & Predictive Robot Intelligence
.PHONY: install test bench base-model upgrade demo docker clean occupancy

install:
	pip install -r requirements.txt

test:            ## fast CI smoke test (<60 s)
	python3 tests/smoke_test.py

bench:           ## full research pipeline: data -> train -> Tests 1-3 -> figures
	python3 scripts/run_pipeline.py

base-model:      ## create the base model only when missing
	python3 scripts/train_base_model.py

upgrade: base-model   ## ensemble uncertainty study: Tests 4-5
	python3 scripts/run_upgrade.py 1 && python3 scripts/run_upgrade.py 2 && python3 scripts/run_upgrade.py 3

demo:            ## regenerate the animated demo GIF
	python3 scripts/make_demo_video.py

docker:
	docker build -t nwm . && docker run --rm nwm python tests/smoke_test.py

clean:
	find . -name __pycache__ -type d -exec rm -rf {} +

occupancy:       ## Stage 1: occupancy-grid prediction + planning (Test 6)
	python3 scripts/run_stage1_occupancy.py 1 && python3 scripts/run_stage1_occupancy.py 2 && python3 scripts/run_stage1_occupancy.py 3
