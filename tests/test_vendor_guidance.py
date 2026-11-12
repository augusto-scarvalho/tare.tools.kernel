"""The owner's choices about the blocks a vendor CLI adds on its own (compute_plane/vendor_guidance.py)."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pytest

from compute_plane import native_cli
from compute_plane.native_cli_codex import ENABLED_FEATURES, REQUIRED_FEATURES
from compute_plane.native_models import NativeTransportInterrupted, sha
from compute_plane.vendor_guidance import catalog, codex_switches, transport_texts, validate
from tare_tools_agent_runtime import ModelFailure
from tests.test_native_models import AnyQualified, execution

PHYSICAL = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
CHAT = {'model': 'fixture', 'effort': 'high',
        'tools': [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a file.', 'parameters': {
            'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'],
            'additionalProperties': False}}}],
        'messages': [{'role': 'system', 'content': 'Role guidance.'}, {'role': 'user', 'content': 'Read a.py.'}]}
CODECS = {'claude-native': ('cli://claude', {'native': True}), 'claude-summary': ('cli://claude', {'summary': True}),
          'claude-toolless': ('cli://claude', {}),
          'claude-structured': ('cli://claude', {'structured': True, 'formatter': True}),
          'agy': ('cli://agy', {'structured': True}),
          'codex': ('cli://codex', {'structured': True, 'native_arguments': True})}
VERSIONS = {'cli://claude': '2.1.280', 'cli://agy': '1.2.7', 'cli://codex': '0.157.0'}


def dispatched(tmp_path, monkeypatch, codec, *, version=None, switches=None, prompt_input=None, calls=None,
               expect=NativeTransportInterrupted, repeat=1, **client):
    """What one call hands its CLI (argv, stdin, the environment it sets, the files it writes) and its receipt,
    stopped before any reply; `expect` is what the call raises (a refusal before dispatch hands nothing)."""
    endpoint, flags = CODECS[codec]
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))  # no global Codex instructions
    seen = {}

    def process(argv, **options):
        if calls is not None:
            calls.append(argv)
        if argv[-1] == '--version':
            return PHYSICAL, (version or VERSIONS[endpoint]).encode(), b''
        if '--help' in argv:
            return PHYSICAL, ' '.join(native_cli.CLI_SWITCHES[endpoint] if switches is None else switches).encode(), b''
        if argv[1:3] == ['features', 'list']:
            return PHYSICAL, '\n'.join(f'{n}  stable  {str("-c" in argv and n in ENABLED_FEATURES).lower()}'
                                       for n in sorted(REQUIRED_FEATURES)).encode(), b''
        if argv[1:3] == ['debug', 'prompt-input']:
            return prompt_input(argv, options)
        root = options['cwd']
        seen.update(root=root, argv=argv[1:], stdin=options['stdin'].decode('utf-8'),
                    env={k: v for k, v in options['env'].items() if os.environ.get(k) != v},
                    files={p.relative_to(root).as_posix(): p.read_text(encoding='utf-8')
                           for p in sorted(root.rglob('*')) if p.is_file()})
        raise RuntimeError('stop before any reply')

    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    cli = native_cli.VendorCliClient(endpoint, executable=str(tmp_path/'cli.exe'), version=VERSIONS[endpoint],
                                     model_catalog={'slug': 'fixture'} if endpoint == 'cli://codex' else None,
                                     qualification=AnyQualified(), **client)
    for _ in range(repeat):
        with pytest.raises(expect) as caught:
            cli.post_observed('', native_cli.cli_request(json.loads(json.dumps(CHAT)), **flags), execution())
    return {**seen, 'receipt': caught.value.transport_observation, 'error': caught.value}


def fingerprint(seen):
    """Digest of a dispatch with machine paths and separators normalized, so it is the same on any machine."""
    def normal(value):
        if isinstance(value, dict):
            return {k: normal(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normal(v) for v in value]
        for path, name in ((seen['root'], '<root>'), (seen['root'].parent, '<tmp>'),
                           (Path(sys.executable), '<python>'), (Path(native_cli.__file__).parent, '<kernel>')):
            for form in (json.dumps(str(path))[1:-1], str(path), path.as_posix()):
                value = value.replace(form, name)
        return re.sub(r'[\\/]+', '/', value)
    value = normal({k: seen[k] for k in ('argv', 'stdin', 'env', 'files')})
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Recorded from the adapter before vendor guidance existed (Kernel cf63a03): with no saved choice every CLI must
# keep receiving exactly these bytes.
BEFORE = {'agy': '3dae06ce2c95d5ac02456be7a0c4abefa84e47d1449b5db86cef907e76922d96',
          'claude-native': '856e71e6797bdf96c8f470fb3946e66316c7024707637a860f9bc8c1003b0694',
          'claude-structured': '9acec21e03908e2c9d814362b06a713fd3e7ebff206ecfb03e26a5a1c2876a72',
          'claude-summary': '53a9f9aa1de77632d6254fe197b02cc5cde28ad70276408ab50198b81192c8b6',
          'claude-toolless': '12ae6937dcbd2944854853e78f64c82e21699d5afac8cf935c506b2932680c4c',
          'codex': '323f3b96baafd112f360d8b66071850e5d262c35d18ac8c9ef0b520eb755afb3'}


@pytest.mark.parametrize('codec', sorted(CODECS))
def test_without_a_choice_every_cli_receives_exactly_what_it_received_before(tmp_path, monkeypatch, codec):
    found = {fingerprint(dispatched(tmp_path/str(i), monkeypatch, codec, **extra))
             for i, extra in enumerate(({}, {'vendor_guidance': None}, {'vendor_guidance': {}}))}
    assert found == {BEFORE[codec]}


CLAUDE_MEASURED = '2.1.322'
WITH_MOD = native_cli.CLI_SWITCHES['cli://claude'] | {'--plugin-dir'}
ALL_CODEX = {item['id']: None for item in catalog()['cli://codex']['items']}


def test_the_catalog_lists_measured_blocks_and_accepts_only_the_controls_each_offers():
    entries = catalog()
    assert set(entries) == {'cli://claude', 'cli://codex', 'cli://agy'}
    for endpoint, entry in entries.items():
        for item in entry['items']:
            assert set(item) == {'id', 'title', 'sent_as', 'controls', 'text'} and item['title'] and item['text']
            assert set(item['controls']) <= {'remove', 'replace'}
            if 'remove' in item['controls']:
                assert validate(endpoint, {item['id']: None}) == {item['id']: None}
    assert entries['cli://agy']['items'] == [] and entries['cli://agy']['measured_version'] is None
    assert validate('cli://claude', {'environment': 'A sandbox.'}) == {'environment': 'A sandbox.'}
    assert validate('cli://codex', None) == {}
    for endpoint, choices in [('cli://claude', {'identity': None}), ('cli://claude', {'user_email': 'other'}),
                              ('cli://codex', {'environment_context': 'other'}), ('cli://claude', {'date': ' '}),
                              ('cli://claude', {'unknown': None}), ('cli://agy', {'anything': None}),
                              ('cli://other', {})]:
        with pytest.raises(ValueError):
            validate(endpoint, choices)
    assert '@' not in json.dumps(entries)  # placeholders, never the measuring account's values


def test_transport_texts_are_what_the_codecs_build():
    texts = {t['id']: t for t in transport_texts()}
    assert texts['claude-native']['text'] == native_cli.NATIVE_INSTRUCTIONS
    assert texts['claude-toolless']['text'] == native_cli.INSTRUCTIONS
    assert texts['agy-action-guidance']['text'] == native_cli.AGY_ACTION_GUIDANCE
    assert texts['codex-structured']['text'].startswith(
        'Tare role guidance for choosing the external decision:\n<role guidance>')
    assert all(t['endpoint'] in catalog() and t['title'] and t['text'] for t in texts.values())


def test_agy_offers_nothing_to_choose(tmp_path):
    with pytest.raises(ValueError):
        native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7',
                                   vendor_guidance={'environment': None})


def test_claude_decisions_carry_the_choices_as_a_mod_and_nothing_else_changes(tmp_path, monkeypatch):
    choices = {'user_email': None, 'environment': None, 'date': 'Today is a working day.'}
    seen = dispatched(tmp_path, monkeypatch, 'claude-native', version=CLAUDE_MEASURED, switches=WITH_MOD,
                      vendor_guidance=choices)
    argv, files = seen['argv'], seen['files']
    at = argv.index('--plugin-dir')
    assert Path(argv[at + 1]) == seen['root']/'tare-guidance'
    source = files.pop('tare-guidance/hooks/register.ts')
    assert 'const dropContext: string[] = ["userEmail"];' in source
    assert ('const attachments: Record<string, string | null> = '
            '{"date": "Today is a working day.", "environment": null};') in source
    assert json.loads(files.pop('tare-guidance/hooks/hooks.json')) == {'modules': ['./register.ts']}
    assert json.loads(files.pop('tare-guidance/.claude-plugin/plugin.json'))['name'] == 'tare-guidance'
    assert fingerprint({**seen, 'argv': argv[:at] + argv[at + 2:]}) == BEFORE['claude-native']  # only the mod
    assert seen['receipt']['vendor_guidance'] == {
        'choices': {'date': 'replace', 'environment': 'remove', 'user_email': 'remove'}, 'sha256': sha(choices)}


def test_the_receipt_says_whether_claude_listed_the_mod_at_init(tmp_path, monkeypatch):
    from tests.test_native_cli import native_claude_rows

    def process(argv, **options):
        if argv[-1] == '--version':
            return PHYSICAL, b'2.1.322 (Claude Code)', b''
        if '--help' in argv:
            return PHYSICAL, ' '.join(WITH_MOD).encode(), b''
        rows, hooks = native_claude_rows(['read'])
        rows[0]['plugins'] = [{'name': 'tare-guidance', 'source': 'tare-guidance@inline'},
                              {'name': 'cc-plugin-telemetry', 'path': 'builtin'}]
        (options['cwd']/'boundary.jsonl').write_text(''.join(json.dumps(h) + '\n' for h in hooks), encoding='utf-8')
        return PHYSICAL, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(native_cli, 'run_pipe_process', process)
    cli = native_cli.VendorCliClient('cli://claude', executable=str(tmp_path/'claude.exe'), version=CLAUDE_MEASURED,
                                     qualification=AnyQualified(), vendor_guidance={'date': None})
    _, receipt = cli.post_observed('', native_cli.cli_request(json.loads(json.dumps(CHAT)), native=True), execution())
    assert receipt['vendor_guidance']['mod_listed_at_init'] is True


@pytest.mark.parametrize('codec', ['claude-summary', 'claude-toolless', 'claude-structured'])
def test_claude_codecs_that_keep_hooks_off_refuse_a_choice_before_dispatch(tmp_path, monkeypatch, codec):
    # Their mod would not run with hooks off, so the choice would be silently ignored (capture of 2026-11-12).
    seen = dispatched(tmp_path, monkeypatch, codec, version=CLAUDE_MEASURED, switches=WITH_MOD,
                      vendor_guidance={'user_email': None}, expect=ModelFailure)
    assert seen['error'].code == 'CLI_GUIDANCE_UNSUPPORTED' and seen['error'].not_started and 'argv' not in seen
    assert seen['receipt']['vendor_guidance']['refused'] == 'hooks_disabled_codec'


@pytest.mark.parametrize('version,switches,code', [('2.1.321', WITH_MOD, 'CLI_GUIDANCE_UNSUPPORTED'),
                                                   ('2.1.322', native_cli.CLI_SWITCHES['cli://claude'],
                                                    'CLI_CAPABILITY_UNAVAILABLE')])
def test_a_claude_that_cannot_load_the_mod_is_refused_before_dispatch(tmp_path, monkeypatch, version, switches, code):
    seen = dispatched(tmp_path, monkeypatch, 'claude-native', version=version, switches=switches,
                      vendor_guidance={'date': None}, expect=ModelFailure)
    assert seen['error'].code == code and seen['error'].not_started and 'argv' not in seen


def rendered_prompt(honoured=True, renamed=(), seen=None):
    """`codex debug prompt-input` of a release that honours the removal switches, or ignores them."""
    def render(argv, options):
        assert options['env']['CODEX_HOME'] == str(options['cwd'])  # no login, no user configuration
        if seen is not None:
            seen.append(argv)
        tags = [tag.replace('<', '<renamed-') if item in renamed else tag
                for item, (key, tag) in codex_switches(ALL_CODEX).items()
                if not (honoured and key + '=false' in argv)]
        return PHYSICAL, json.dumps([{'type': 'message', 'role': 'developer', 'content': [
            {'type': 'input_text', 'text': tag + ' vendor text'}]} for tag in tags]).encode(), b''
    return render


def test_codex_removals_are_proved_once_per_release_then_sent_as_switches(tmp_path, monkeypatch):
    choices, shown = {'environment_context': None, 'skills_instructions': None}, []
    seen = dispatched(tmp_path, monkeypatch, 'codex', vendor_guidance=choices, prompt_input=rendered_prompt(seen=shown),
                      repeat=2)
    assert len(shown) == 2  # without the switches, then with them; the second call reuses the proof
    argv = seen['argv']
    switches = [i for i, a in enumerate(argv)
                if a in ('include_environment_context=false', 'skills.include_instructions=false')]
    assert len(switches) == 2 and all(argv[i - 1] == '-c' for i in switches)
    plain = [a for i, a in enumerate(argv) if i not in switches and i + 1 not in switches]
    files = {k: v for k, v in seen['files'].items() if not k.startswith('prompt-input/')}
    assert fingerprint({**seen, 'argv': plain, 'files': files}) == BEFORE['codex']  # only the switches
    assert seen['receipt']['vendor_guidance']['choices'] == {'environment_context': 'remove',
                                                             'skills_instructions': 'remove'}


@pytest.mark.parametrize('honoured,renamed', [(False, ()), (True, ('collaboration_mode',))])
def test_a_codex_removal_its_release_does_not_prove_is_refused_before_dispatch(tmp_path, monkeypatch, honoured,
                                                                              renamed):
    seen = dispatched(tmp_path, monkeypatch, 'codex', vendor_guidance={'collaboration_mode': None},
                      prompt_input=rendered_prompt(honoured, renamed), expect=ModelFailure)
    assert seen['error'].code == 'CLI_GUIDANCE_UNSUPPORTED' and seen['error'].not_started and 'argv' not in seen
    assert seen['receipt']['vendor_guidance']['unproved'] == ['collaboration_mode']
