#!/usr/bin/env bash
# Download the complete training curves used in Figure 2.
#   bash scripts/download_curves.sh [URL]
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${1:-https://github.com/xiaosly/masafe-power-coordination/releases/download/v1.0/masafe_training_curves.zip}
curl -fL -o masafe_training_curves.zip "$URL"
unzip -oq masafe_training_curves.zip
