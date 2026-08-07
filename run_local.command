#!/bin/zsh
set -e

cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  uv sync --all-groups
fi

exec .venv/bin/python -m streamlit run streamlit_app.py \
  --server.port 8765 \
  --browser.gatherUsageStats false
