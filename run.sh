#!/usr/bin/env bash
# Convenience launcher
set -e
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
exec streamlit run app.py "$@"
