"""A CLI call is judged by the silence of its stdout when the host gives a window, not by a total clock."""
import json
import math
import sys
import time

import pytest

from compute_plane import native_cli
from compute_plane.native_models import NativeTransportInterrupted
from tests.test_native_cli import structured_claude_rows
from tests.test_native_models import AnyQualified, execution

PHYSICAL = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
FINAL = ''.join(json.dumps(row) + '\n' for row in structured_claude_rows())


def client(tmp_path, monkeypatch, script, endpoint='cli://claude', **options):
    """Everything around the call is answered locally; the call itself runs `script` as a real supervised process."""
    real = native_cli.run_pipe_process

    def process(argv, **kwargs):
        if '--version' in argv:
            return PHYSICAL, b'2.1.282', b''
        if '--help' in argv:
            return PHYSICAL, ' '.join(native_cli.CLI_SWITCHES[endpoint]).encode(), b''
        return real([sys.executable, '-u', '-c', script], **kwargs)
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    return native_cli.VendorCliClient(endpoint, executable=str(tmp_path/'claude.exe'), version='2.1.280',
                                      qualification=AnyQualified(), **options)


BODY = native_cli.cli_request({'model': 'fixture', 'messages': [], 'tools': [{'function': {'name': 'read', 'parameters': {}}}]},
                              structured=True, formatter=True)
STREAMING = ('import time,sys\nfor _ in range(8):\n print(\'{"type":"system"}\');time.sleep(.4)\n'
             'sys.stdout.write(%r)' % FINAL)
SILENT = 'import time\nprint(\'{"type":"system"}\')\nwhile True: pass\n'  # busy: its CPU time grows while it says nothing


def test_a_call_that_keeps_writing_is_never_stopped_by_a_clock_of_its_own(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, STREAMING, timeout_seconds=None, silence_seconds=2)
    assert math.isinf(c.timeout_seconds)  # no total bound
    started = time.monotonic()
    response, receipt = c.post_observed('', BODY, execution())
    assert time.monotonic() - started > 3.2 and response and not receipt.get('stalled')


def test_a_silent_call_is_stopped_at_the_window_with_its_facts_even_while_its_cpu_works(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, SILENT, timeout_seconds=None, silence_seconds=2)
    started = time.monotonic()
    with pytest.raises(NativeTransportInterrupted) as caught:
        c.post_observed('', BODY, execution())
    receipt = caught.value.transport_observation
    assert str(caught.value) == 'CLI_SILENT_EXCHANGE' and time.monotonic() - started < 15
    assert receipt['stalled'] is True and receipt['stopped_by_observer'] is True and receipt['timeout'] is False
    assert 2000 <= receipt['idle_ms'] < 6000 and receipt['silence_seconds'] == 2
    assert receipt['cpu_ms'] > 0 and receipt['cleanup_verified'] is True


def test_the_silence_window_is_measured_from_the_last_byte_not_from_the_start():
    now = [0.0]
    silence = native_cli._Silence(10, clock=lambda: now[0])
    now[0] = 9
    assert silence({}) is False
    silence.feed(b'x')
    now[0] = 18.9
    assert silence({}) is False
    now[0] = 19
    assert silence({}) is True and silence.idle_ms == 10000


@pytest.mark.parametrize('endpoint', ['cli://codex', 'cli://agy'])
def test_a_cli_that_prints_only_when_the_reply_is_whole_keeps_its_total_bound(tmp_path, endpoint):
    c = native_cli.VendorCliClient(endpoint, executable=str(tmp_path/'x.exe'), version='1.2.7',
                                   silence_seconds=600, timeout_seconds=None, model_catalog={'slug': 'x'} if 'codex' in endpoint else None)
    assert c.silence_seconds is None and c.timeout_seconds == 120


def test_without_a_window_the_bound_is_what_it_was(tmp_path):
    kwargs = dict(executable=str(tmp_path/'claude.exe'), version='2.1.280')
    assert native_cli.VendorCliClient('cli://claude', timeout_seconds=None, **kwargs).timeout_seconds == 120
    assert native_cli.VendorCliClient('cli://claude', timeout_seconds=300, **kwargs).timeout_seconds == 300
    for bad in (math.inf, math.nan, 0, -1):
        with pytest.raises(ValueError):
            native_cli.VendorCliClient('cli://claude', timeout_seconds=bad, **kwargs)
    for bad in (0, -1, math.inf, math.nan, '600', True):
        with pytest.raises(ValueError):
            native_cli.VendorCliClient('cli://claude', silence_seconds=bad, **kwargs)
    # a window and a total may both be given: the first to trip wins
    assert native_cli.VendorCliClient('cli://claude', timeout_seconds=300, silence_seconds=60, **kwargs).timeout_seconds == 300
