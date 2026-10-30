#!/usr/bin/env bash
# Sets up the virtualenv on first run, then starts Teleport Cam. Extra args are passed through.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m teleport_cam "$@"
