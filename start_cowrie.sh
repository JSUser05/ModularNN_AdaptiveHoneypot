#!/bin/bash

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_COWRIE="${ROOT_DIR}/cowrie/bin/cowrie"
export PYTHONPATH="${ROOT_DIR}/cowrie/src:${PYTHONPATH:-}"

if [ ! -x "${LOCAL_COWRIE}" ]; then
  echo "Local cowrie launcher not found: ${LOCAL_COWRIE}"
  exit 1
fi

"${LOCAL_COWRIE}" restart
