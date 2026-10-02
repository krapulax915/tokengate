.PHONY: run test bench sim demo ab seed clean

PY ?= python

# Start the gateway (serves the dashboard at /). Port 8000 is intentionally
# avoided: it collides with other local services on some machines.
run:
	$(PY) -m uvicorn dietgate.main:app --host 127.0.0.1 --port 8080

# Run the test suite
test:
	$(PY) -m pytest

# Run the built-in simulator against a running gateway (2000 tasks)
sim:
	$(PY) -m sim.traffic --scenario default --url http://127.0.0.1:8080

# A/B comparison: static (baseline) vs thompson
ab:
	$(PY) -m sim.traffic --scenario default --ab --url http://127.0.0.1:8080

# Overhead benchmark (starts its own gateway with zero-latency mocks)
bench:
	$(PY) bench/overhead.py

# One-command demo: gateway + simulator + dashboard
demo:
	$(PY) scripts/run_demo.py

# Pre-fill the DB with a realistic learning curve
seed:
	$(PY) scripts/seed_demo.py

clean:
	rm -rf data/*.db* bench/results/*.json bench/results/*.md
