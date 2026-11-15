"""A process tree is judged by its activity (output and CPU time), not only by a wall clock."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from compute_plane import managed_process
from compute_plane.managed_process import run_pipe_process, run_process
from compute_plane.native_capabilities import NativeCapabilities
from tare_tools_agent_runtime import BindingRef, CancellationToken, EffectRequest, ExecutionControl, ToolCall

pytestmark = pytest.mark.skipif(os.name != 'nt' and not sys.platform.startswith('linux'),
                                reason='tree CPU time is measured on Windows and Linux only')

BUSY = 'import time\nend = time.monotonic() + {}\nwhile time.monotonic() < end: pass'
SILENT = 'import time; time.sleep({})'
CHATTY = 'import time\nfor _ in range({}): print("tick", flush=True); time.sleep(.4)'
GRANDCHILD = 'import subprocess, sys; subprocess.run([sys.executable, "-I", "-c", {!r}])'
RUNNERS = pytest.mark.parametrize('runner', ['run_process', 'run_pipe_process'])


def run(runner, tmp_path, code, **options):
    argv = [sys.executable, '-I', '-c', code]
    started = time.monotonic()
    if runner == 'run_process':
        receipt = run_process(argv, cwd=tmp_path, dest=tmp_path/'out', **options)
    else:
        receipt, _, _ = run_pipe_process(argv, cwd=tmp_path, **options)
    return receipt, time.monotonic() - started


@RUNNERS
def test_a_busy_child_that_prints_nothing_outlives_the_idle_limit(tmp_path, runner):
    receipt, _ = run(runner, tmp_path, BUSY.format(3), timeout=30, idle_timeout=1.5)
    assert receipt['exit_code'] == 0 and receipt['cleanup_verified']
    assert receipt['stalled'] is False and receipt['timeout'] is False
    assert receipt['activity'] == 'cpu+output' and receipt['cpu_ms'] > 1000 and receipt['output_bytes'] == 0


@RUNNERS
def test_a_silent_sleeping_child_is_stopped_at_its_idle_limit(tmp_path, runner):
    receipt, wall = run(runner, tmp_path, SILENT.format(60), timeout=30, idle_timeout=1.5)
    assert receipt['stalled'] is True and receipt['timeout'] is False and receipt['exit_code'] is None
    assert receipt['cleanup_verified'] and receipt['idle_ms'] >= 1500 and wall < 10


@RUNNERS
def test_a_child_that_prints_while_sleeping_is_not_stalled(tmp_path, runner):
    receipt, _ = run(runner, tmp_path, CHATTY.format(7), timeout=30, idle_timeout=1.5)
    assert receipt['exit_code'] == 0 and receipt['stalled'] is False and receipt['output_bytes'] > 0


@RUNNERS
def test_the_tree_is_measured_when_only_a_grandchild_works(tmp_path, runner):
    receipt, _ = run(runner, tmp_path, GRANDCHILD.format(BUSY.format(3)), timeout=30, idle_timeout=1.5)
    assert receipt['exit_code'] == 0 and receipt['stalled'] is False and receipt['cpu_ms'] > 1000


@RUNNERS
@pytest.mark.parametrize('code', [BUSY.format(30), SILENT.format(30)], ids=['busy', 'silent'])
def test_the_ceiling_still_stops_a_child_before_a_longer_idle_limit(tmp_path, runner, code):
    receipt, wall = run(runner, tmp_path, code, timeout=1.5, idle_timeout=10)
    assert receipt['timeout'] is True and receipt['stalled'] is False and receipt['cleanup_verified'] and wall < 8


@RUNNERS
def test_progress_is_sampled_while_it_runs_and_a_failing_observer_breaks_nothing(tmp_path, runner):
    samples = []
    receipt, _ = run(runner, tmp_path, CHATTY.format(6), timeout=30, on_progress=samples.append)
    assert receipt['exit_code'] == 0 and receipt['progress_failures'] == 0 and receipt['stalled'] is False
    assert len(samples) >= 2 and all(set(s) == {'elapsed_ms', 'idle_ms', 'cpu_ms', 'output_bytes'} for s in samples)
    assert all(a['elapsed_ms'] < b['elapsed_ms'] for a, b in zip(samples, samples[1:]))
    def fail(sample): raise RuntimeError('observer failure')
    receipt, _ = run(runner, tmp_path, CHATTY.format(6), timeout=30, on_progress=fail)
    assert receipt['exit_code'] == 0 and receipt['cleanup_verified'] and receipt['progress_failures'] >= 2


@RUNNERS
def test_an_observer_that_answers_true_stops_the_tree_and_no_other_answer_does(tmp_path, runner):
    # A host that asks someone before stopping a silent tree gives no idle limit: the tree runs while it decides.
    receipt, wall = run(runner, tmp_path, SILENT.format(30), timeout=30, on_progress=lambda sample: sample['idle_ms'] > 1500)
    assert receipt['stalled'] is True and receipt['stopped_by_observer'] is True and receipt['timeout'] is False
    assert receipt['cleanup_verified'] and wall < 15
    receipt, _ = run(runner, tmp_path, SILENT.format(3), timeout=30, on_progress=lambda sample: 'stop')
    assert receipt['exit_code'] == 0 and receipt['stalled'] is False and 'stopped_by_observer' not in receipt


@RUNNERS
def test_without_a_cpu_measurement_the_idle_limit_stops_nothing(tmp_path, runner, monkeypatch):
    monkeypatch.setattr(managed_process, '_tree_cpu_ms', lambda job, pgid: None)
    receipt, _ = run(runner, tmp_path, SILENT.format(3), timeout=30, idle_timeout=1)
    assert receipt['exit_code'] == 0 and receipt['stalled'] is False
    assert receipt['activity'] == 'output' and receipt['cpu_ms'] is None and receipt['idle_ms'] >= 2000


def test_without_the_new_parameters_the_receipts_keep_their_fields(tmp_path):
    receipt, _ = run('run_process', tmp_path, 'pass', timeout=30)
    assert set(receipt) == {'exit_code', 'timeout', 'cancelled', 'output_limit', 'cleanup_verified', 'containment',
                            'pid', 'descendants_at_exit', 'wall_ms'}
    receipt, _ = run('run_pipe_process', tmp_path, 'pass', timeout=30)
    assert set(receipt) == {'exit_code', 'timeout', 'cancelled', 'output_limit', 'cleanup_verified', 'containment',
                            'started', 'pid', 'wall_ms'}


@RUNNERS
@pytest.mark.parametrize('options', [{'idle_timeout': 0}, {'idle_timeout': True}, {'idle_timeout': math.nan},
                                     {'idle_timeout': '5'}, {'on_progress': 1}])
def test_invalid_supervision_is_refused_before_a_process_starts(tmp_path, runner, options, monkeypatch):
    monkeypatch.setattr(managed_process.subprocess, 'Popen', lambda *a, **kw: pytest.fail('process started'))
    with pytest.raises(ValueError):
        run(runner, tmp_path, 'pass', timeout=30, **options)


def capabilities(tmp_path, seconds=20, **binding):
    root, scratch = tmp_path/'repo', tmp_path/'custody'
    root.mkdir(); scratch.mkdir()
    ref = BindingRef('work', 'binding', 'owner', 1, 'token')
    control = ExecutionControl(ref, CancellationToken(), time.monotonic()+30)
    tools = NativeCapabilities(session_id='s', binding=ref, repositories={'repo': str(root)}, writable={'repo': ['.']},
        verifiers={'unit': {'repository': 'repo', 'argv': [sys.executable, '-I', '-c', SILENT.format(60)]}},
        scratch_root=scratch, capabilities=['command', 'verify'], specgraph_edit_context=False)
    class Mediation:
        def execute(self, facts, effect): return effect()
    tools.bind_command_mediation(Mediation())
    environment = {name: os.environ[name] for name in ('PATH', 'SYSTEMROOT') if name in os.environ}
    tools.bind_command_confinement(subprocess.Popen, environment, seconds, **binding)
    return tools, control


def call(tools, control, capability, **args):
    return tools.execute(EffectRequest('s', 's/effect/1', ToolCall('t', capability, json.dumps(args))), control)


@pytest.mark.parametrize('capability,binding,status', [
    ('verify', {'verification_idle_seconds': 1.5}, 'unknown'),
    ('command', {'idle_seconds': 1.5}, 'unknown'),
    ('command', {'idle_seconds': 1.5, 'staged_effects': True}, 'failed')])
def test_a_bound_idle_limit_stops_a_silent_verify_or_command_and_says_so(tmp_path, capability, binding, status):
    tools, control = capabilities(tmp_path, **binding)
    args = {'name': 'unit'} if capability == 'verify' else {'repository': 'repo',
                                                             'argv': [sys.executable, '-I', '-c', SILENT.format(60)]}
    result = call(tools, control, capability, **args)
    output = json.loads(result.output_json)
    assert result.status == status and 'showed no activity' in output['reason']
    assert output['reason'].startswith('Staged command' if 'staged_effects' in binding else 'The process')
    process = output['process']
    assert process['stalled'] is True and process['timeout'] is False and process['cleanup_verified']
    assert process['idle_ms'] >= 1500 and process['cpu_ms'] is not None


@pytest.mark.parametrize('unbounded', [True, False])
def test_an_unbounded_command_with_an_idle_limit_runs_past_its_ceiling_and_reports_progress(tmp_path, unbounded):
    seen = []
    tools, control = capabilities(tmp_path, seconds=1, idle_seconds=1.5, unbounded_command=unbounded,
                                  on_progress=lambda effect, sample: seen.append((effect, sample)))
    result = call(tools, control, 'command', repository='repo', argv=[sys.executable, '-I', '-c', CHATTY.format(6)])
    output = json.loads(result.output_json)
    if unbounded:
        assert result.status == 'succeeded' and output['returncode'] == 0 and output['output'].count('tick') == 6
        assert output['process']['wall_ms'] > 2000 and output['process']['timeout'] is False
        assert seen and all(effect == 's/effect/1' and 'elapsed_ms' in sample for effect, sample in seen)
    else:  # The ceiling keeps its meaning and its receipt, with no stall text.
        assert result.status == 'unknown' and output['process']['timeout'] is True and 'reason' not in output


@pytest.mark.parametrize('binding,verify,command', [
    ({}, set(), set()),
    ({'verification_idle_seconds': 5}, {'idle_timeout'}, set()),
    ({'idle_seconds': 5, 'on_progress': lambda effect, sample: None}, {'on_progress'}, {'idle_timeout', 'on_progress'})])
def test_only_what_the_host_bound_reaches_the_process(tmp_path, monkeypatch, binding, verify, command):
    seen = []
    def runner(argv, **options):
        seen.append(set(options) & {'idle_timeout', 'on_progress'})
        Path(options['dest'], 'stdout.txt').write_bytes(b'')
        Path(options['dest'], 'stderr.txt').write_bytes(b'')
        return dict(exit_code=0, cleanup_verified=True, cancelled=False, timeout=False, output_limit=False)
    monkeypatch.setattr('compute_plane.managed_process.run_process', runner)
    tools, control = capabilities(tmp_path, **binding)
    assert call(tools, control, 'verify', name='unit').status == 'succeeded'
    assert call(tools, control, 'command', repository='repo', argv=['fixture']).status == 'succeeded'
    assert seen == [verify, command]


@pytest.mark.parametrize('binding', [{'unbounded_command': 'yes'}, {'idle_seconds': 0},
                                     {'verification_idle_seconds': math.inf}, {'on_progress': 1}])
def test_invalid_supervision_binding_is_refused(tmp_path, binding):
    with pytest.raises(ValueError):
        capabilities(tmp_path, **binding)


def test_any_growth_of_cpu_time_is_activity_however_small():
    # A tree bound by I/O: 5 ms of CPU a second, no output. Under a 1% threshold it read as silent for 300 s.
    clock, cpu = [1000.0], [0.0]
    watch = managed_process._Activity(300, None)
    real = time.monotonic
    try:
        time.monotonic = lambda: clock[0]
        result = {}
        watch.start(result, lambda: cpu[0])
        for _ in range(300):
            clock[0] += 1; cpu[0] += 5
            assert watch.stalled(0) is False
        assert result['idle_ms'] == 0 and result['cpu_ms'] == 1500
        for _ in range(299):
            clock[0] += 1
            assert watch.stalled(0) is False  # the CPU time stopped moving: silence starts counting
        clock[0] += 1
        assert watch.stalled(0) is True
    finally:
        time.monotonic = real
