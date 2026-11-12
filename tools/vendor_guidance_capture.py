"""Read what a vendor CLI adds to tare's requests now, spending no tokens (docs/VENDOR_GUIDANCE.md).

  python tools/vendor_guidance_capture.py claude|codex <executable> [choices]

`choices` is a JSON object of item ids from a reading, e.g. '{"context:userEmail": null}': the CLI is read with
them applied, to see that they take. The reading is compute_plane.vendor_guidance.read; this prints each item's id,
controls, sha256 and text (machine values already placeholders) and writes nothing.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_plane.vendor_guidance import read  # noqa: E402

if __name__ == '__main__':
    vendor, executable, *rest = sys.argv[1:]
    started = time.monotonic()
    value = read('cli://' + vendor, executable, choices=json.loads(rest[0]) if rest else None)
    print(f'{value["cli"]} {value["cli_version"]} read in {time.monotonic() - started:.1f} s')
    for item in value['items']:
        digest = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
        print(f'\n--- {item["id"]} [{", ".join(item["controls"]) or "read only"}] sha256 {digest[:16]} · '
              f'{item["sent_as"]}\n{item["text"]}')
