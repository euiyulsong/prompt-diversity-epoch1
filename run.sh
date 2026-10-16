#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python run.py "$@"
python -m pip freeze > results_environment.txt
