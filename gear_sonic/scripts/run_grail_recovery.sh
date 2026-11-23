#!/usr/bin/env bash
# Reuse the carry launcher with randomized failed-grasp starts.
set -eo pipefail

exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_grail_carry.sh" --recovery
