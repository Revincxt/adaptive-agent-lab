.PHONY: install lint typecheck test check build web-install web-build web-test demo clean

install:
	python -m pip install -e '.[dev]'

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest --cov=adaptive_agent_lab --cov-report=term-missing

check: lint typecheck test

build:
	python -m build

web-install:
	pnpm --dir web install --frozen-lockfile

web-build:
	pnpm --dir web run build

web-test:
	pnpm --dir web run test

demo:
	aal run --agent replanning --scenario scenarios/small/dynamic-demo.json --seed 42

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
