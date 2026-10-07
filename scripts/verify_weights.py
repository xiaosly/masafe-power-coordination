"""Check pretrained/ against the SHA-256 hashes in pretrained/manifest.json."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "pretrained"
manifest = json.loads((root / "manifest.json").read_text())
bad = [name for name, digest in manifest["sha256"].items()
       if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
if bad:
    raise SystemExit(f"hash mismatch: {bad}")
print(f"{len(manifest['sha256'])} actor files verified in {root}")
