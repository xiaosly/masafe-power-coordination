#!/usr/bin/env bash
# Download the trained policies of the proposed method and check their SHA-256 hashes.
#   bash scripts/download_weights.sh [URL]
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${1:-https://github.com/xiaosly/masafe-power-coordination/releases/download/v1.0/masafe_proposed_weights.zip}
curl -fL -o masafe_proposed_weights.zip "$URL"
unzip -oq masafe_proposed_weights.zip
python scripts/verify_weights.py
