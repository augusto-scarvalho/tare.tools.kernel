"""Real audit transport subprocesses with legacy driver imports forbidden."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode', ['success', 'timeout', 'cancel'])
def test_audit_process_lifecycle_does_not_import_aider(tmp_path, mode):
    script = tmp_path / 'probe.py'
    script.write_text('''import json, sys, time
from pathlib import Path
class NoAider:
    def find_spec(self, fullname, *args):
        if fullname == 'aider' or fullname.startswith('aider.') or fullname == 'compute_plane.aider_driver':
            raise AssertionError('legacy import: ' + fullname)
sys.meta_path.insert(0, NoAider())
from compute_plane.audit_driver import ClaudeAuditDriver
driver = ClaudeAuditDriver([sys.executable])
mode = sys.argv[1]
command = [sys.executable, '-c', "print('native')" if mode == 'success' else 'import time; time.sleep(30)']
start = time.monotonic()
try:
    code, raw = driver.run(command, root=Path(__file__).parent, prompt=b'',
                           timeout=5 if mode == 'success' else .2,
                           cancelled=(lambda: True) if mode == 'cancel' else None)
    assert mode == 'success' and code == 0 and raw.strip() == b'native'
except TimeoutError:
    assert mode == 'timeout'
except InterruptedError:
    assert mode == 'cancel'
assert time.monotonic() - start < 5
print(json.dumps({'status':'PASS','mode':mode}))
''', encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                                       env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-B', str(script), mode], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'status': 'PASS', 'mode': mode}
