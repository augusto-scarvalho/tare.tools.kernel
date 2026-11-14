"""Read what a vendor CLI adds to tare's requests (docs/VENDOR_GUIDANCE.md).

  python tools/vendor_guidance_capture.py claude|codex <executable> [choices]          no tokens
  python tools/vendor_guidance_capture.py real claude|agy <executable> <model> [choices] one small real call
  python tools/vendor_guidance_capture.py models <claude executable> <alias>...         no tokens

`choices` is a JSON object of item ids from a reading, e.g. '{"context:userEmail": null}': the CLI is read with
them applied, to see that they take. The readings are compute_plane.vendor_guidance.read and read_real; this
prints each item's id, controls, sha256 and text (machine values already placeholders) and writes nothing. `models`
prints the model id the installed Claude Code resolves each family alias to (claude_models), or None.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_plane.vendor_guidance import claude_models, read, read_real  # noqa: E402

if __name__ == '__main__':
    args = sys.argv[1:]
    started = time.monotonic()
    if args[0] == 'models':
        value = claude_models(args[1], args[2:])
        print(f'Claude Code {value["cli_version"]} read in {time.monotonic() - started:.1f} s')
        for alias, model in value['models'].items():
            print(f'{alias} -> {model}')
        sys.exit()
    if args[0] == 'real':  # spends tokens: only when asked by name
        _, vendor, executable, model, *rest = args
        value = read_real('cli://' + vendor, executable, model=model, choices=json.loads(rest[0]) if rest else None)
    else:
        vendor, executable, *rest = args
        value = read('cli://' + vendor, executable, choices=json.loads(rest[0]) if rest else None)
    print(f'{value["cli"]} {value["cli_version"]} read in {time.monotonic() - started:.1f} s'
          + (f' · usage {value["usage"]}' if 'usage' in value else '')
          + (f' · left out {value["left_out"]}' if value.get('left_out') else ''))
    for item in value['items']:
        digest = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
        local = f' · zero-token reading: {item["local"]}' if 'local' in item else ''
        print(f'\n--- {item["id"]} [{", ".join(item["controls"]) or "read only"}] sha256 {digest[:16]} · '
              f'{item["sent_as"]}{local}\n{item["text"]}')
