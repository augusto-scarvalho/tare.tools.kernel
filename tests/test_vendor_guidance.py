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
from compute_plane import vendor_guidance
from compute_plane.vendor_guidance import (CODEX_SWITCHES, claude_items, codex_items, codex_switches, normalized,
                                           transport_texts, validate)
from tare_tools_agent_runtime import ModelFailure
from tests.test_native_models import AnyQualified, execution

PHYSICAL = dict(started=True, cleanup_verified=True, exit_code=0, cancelled=False, timeout=False, output_limit=False)
CHAT = {'model': 'fixture', 'effort': 'high',
        'tools': [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a file.', 'parameters': {
            'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'],
            'additionalProperties': False}}}],
        'messages': [{'role': 'system', 'content': 'Role guidance.'}, {'role': 'user', 'content': 'Read a.py.'}]}
# claude-plain is how a Claude call without tools goes (generate_text: heads-up notes, plain answers): the native codec
# with an empty catalog, so Claude Code runs with hooks off, like summaries and the rollback codecs.
CODECS = {'claude-native': ('cli://claude', {'native': True}), 'claude-summary': ('cli://claude', {'summary': True}),
          'claude-plain': ('cli://claude', {'native': True}), 'claude-toolless': ('cli://claude', {}),
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
            chat = json.loads(json.dumps({**CHAT, 'tools': []} if codec == 'claude-plain' else CHAT))
            cli.post_observed('', native_cli.cli_request(chat, **flags), execution())
    return {**seen, 'receipt': caught.value.transport_observation, 'error': caught.value}


def fingerprint(seen):
    """Digest of a dispatch with machine paths and separators normalized, so it is the same on any machine."""
    def normal(value):
        if isinstance(value, dict):
            return {k: normal(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normal(v) for v in value]
        places = ((seen['root'], '<root>'), (seen['root'].parent, '<tmp>'), (Path(sys.executable), '<python>'),
                  (Path(native_cli.__file__).parent, '<kernel>'))
        for path, name in sorted(places, key=lambda p: -len(str(p[0]))):  # a checkout may sit in the temp folder
            for form in (json.dumps(str(path))[1:-1], str(path), path.as_posix()):
                value = value.replace(form, name)
        return re.sub(r'[\\/]+', '/', value)
    value = normal({k: seen[k] for k in ('argv', 'stdin', 'env', 'files')})
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Recorded from the adapter before vendor guidance existed (Kernel cf63a03): with no saved choice every CLI must
# keep receiving exactly these bytes.
BEFORE = {'agy': '3dae06ce2c95d5ac02456be7a0c4abefa84e47d1449b5db86cef907e76922d96',
          'claude-native': '856e71e6797bdf96c8f470fb3946e66316c7024707637a860f9bc8c1003b0694',
          'claude-plain': '3ff2f1e8baa5e93f595d887e7d9085d545a9624c9219465f450143599da35e5d',
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
ALL_CODEX = dict.fromkeys(CODEX_SWITCHES)
READING_BODY = native_cli.cli_request({'model': 'claude-sonnet-5-5', 'tools': [], 'messages': [
    {'role': 'user', 'content': vendor_guidance.READING_PROMPT}]}, native=True)


def test_choices_are_checked_by_the_shape_of_their_id():
    later = {'context:userEmail': None, 'attachment:environment': 'A sandbox.', 'attachment:added_later': None}
    assert validate('cli://claude', later) == later  # a block a newer CLI adds is controlled like its kind
    assert validate('cli://codex', {'environment_context': None}) == {'environment_context': None}
    assert validate('cli://agy', None) == {}
    for endpoint, choices in [('cli://claude', {'system:2': None}), ('cli://claude', {'context:userEmail': 'other'}),
                              ('cli://claude', {'message:1': None}), ('cli://claude', {'attachment:date': ' '}),
                              ('cli://claude', {'attachment:': None}), ('cli://claude', {'user_email': None}),
                              ('cli://codex', {'environment_context': 'other'}), ('cli://codex', {'block:5': None}),
                              ('cli://agy', {'anything': None}), ('cli://other', {})]:
        with pytest.raises(ValueError):
            validate(endpoint, choices)


def recorded_claude(root, email='someone@example.com', day='2026-11-12'):
    """A request as the reading's recorder keeps it: Claude Code 2.1.322's shape with the reading mod's marks."""
    from compute_plane.native_cli_claude import input_record
    context = ('<system-reminder>\n<<tare attachment session_context>>As you answer, use this context:\n'
               '# userEmail\n<<tare context userEmail>>The user\'s email address is ' + email + '.<</tare context>>'
               '\n\nAttached by Claude Code.<</tare attachment>>\n</system-reminder>\n')
    trailing = ('<<tare attachment environment>># Environment\n - Primary working directory: ' + str(root) +
                '\n - OS Version: Windows ' + day + '<</tare attachment>>\n\n<<tare attachment date>>'
                'Today\'s date is ' + day + '.<</tare attachment>>')
    return {'system': [{'text': 'x-anthropic-billing-header: cc_version=2.1.322.b46; cc_entrypoint=sdk-cli;'},
                       {'text': 'You are a Claude agent.'}, {'text': READING_BODY['system']}],
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': context},
                                                       *input_record(READING_BODY)['message']['content']]},
                         {'role': 'system', 'content': [{'type': 'text', 'text': trailing}]}],
            'tools': [{'name': 'mcp__tare__read', 'description': 'tare\'s own'},
                      {'name': 'StructuredOutput', 'description': 'Return the result.'}]}


def test_a_claude_reading_finds_each_block_by_its_mark_and_two_readings_agree(tmp_path):
    first = claude_items(recorded_claude(tmp_path/'one'), READING_BODY, [tmp_path/'one'])
    later = claude_items(recorded_claude(tmp_path/'two', 'other@example.org', '2027-01-30'), READING_BODY,
                         [tmp_path/'two'])
    assert first == later  # machine and day values are placeholders
    items = {item['id']: item for item in first}
    assert list(items) == ['system:1', 'system:2', 'context:userEmail', 'attachment:session_context', 'message:1',
                           'attachment:environment', 'attachment:date', 'tool:StructuredOutput']
    assert items['context:userEmail']['text'] == "The user's email address is <account email>."
    assert items['context:userEmail']['controls'] == ('remove',)
    assert '# userEmail\n<context userEmail>' in items['attachment:session_context']['text']
    assert items['attachment:environment']['controls'] == ('remove', 'replace')
    assert (' - Primary working directory: <working directory>\n - OS Version: <OS version>'
            in items['attachment:environment']['text'])
    assert items['system:1']['text'] == 'x-anthropic-billing-header: cc_version=<CLI version>; cc_entrypoint=sdk-cli;'
    assert all(items[k]['controls'] == () for k in ('system:1', 'system:2', 'message:1', 'tool:StructuredOutput'))
    assert items['attachment:environment']['sent_as'] == "system message after tare's records"


def test_a_codex_reading_lists_every_block_but_tares_by_its_leading_tag(tmp_path):
    rendered = json.dumps([
        {'type': 'message', 'role': 'developer', 'content': [
            {'type': 'input_text', 'text': '<skills_instructions>\r\n- `r0` = `' + str(tmp_path/'skills') + '`\r\n'},
            {'type': 'input_text', 'text': '<a_new_block>\nNew text.'}]},
        {'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': '<environment_context>\n  <cwd>' + str(tmp_path/'home') + '</cwd>\n'
                                           '  <current_date>2026-11-12</current_date>\n'
                                           '  <timezone>America/Sao_Paulo</timezone>\n</environment_context>'},
            {'type': 'input_text', 'text': vendor_guidance.READING_PROMPT}]}])
    items = {item['id']: item for item in codex_items(rendered, [tmp_path])}
    assert list(items) == ['skills_instructions', 'a_new_block', 'environment_context']
    assert items['skills_instructions']['text'] == '<skills_instructions>\n- `r0` = `<skills folder>`\n'
    assert items['environment_context']['text'] == (
        '<environment_context>\n  <cwd><working directory></cwd>\n  <current_date><date></current_date>\n'
        '  <timezone><time zone></timezone>\n</environment_context>')
    assert items['skills_instructions']['controls'] == ('remove',) and items['a_new_block']['controls'] == ()


def test_reading_claude_runs_tares_command_against_a_local_recorder_and_stops_at_the_first_request(
        tmp_path, monkeypatch):
    import urllib.error
    import urllib.request
    calls = []

    def process(argv, **options):
        if argv[-1] == '--version':
            return PHYSICAL, b'2.1.322 (Claude Code)', b''
        calls.append(argv)
        root = options['cwd']
        assert 'const reading: boolean = true;' in (root/'tare-guidance/hooks/register.ts').read_text(encoding='utf-8')
        settings = json.loads(argv[argv.index('--settings') + 1])
        assert 'disableAllHooks' not in settings and settings['enabledPlugins'] == {}
        request = urllib.request.Request(options['env']['ANTHROPIC_BASE_URL'] + '/v1/messages?beta=true',
                                         data=json.dumps(recorded_claude(root)).encode(),
                                         headers={'Authorization': 'Bearer not-kept'}, method='POST')
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=10)
        assert refused.value.code == 400 and options['completed']() is True
        return PHYSICAL | {'protocol_complete': True}, b'', b''
    monkeypatch.setattr(vendor_guidance, 'run_pipe_process', process)
    value = vendor_guidance.read('cli://claude', str(tmp_path/'claude.exe'))
    assert value['cli_version'] == '2.1.322' and len(calls) == 1
    assert 'not-kept' not in json.dumps(value) and 'context:userEmail' in {i['id'] for i in value['items']}


def test_an_old_claude_cannot_be_read(tmp_path, monkeypatch):
    monkeypatch.setattr(vendor_guidance, 'run_pipe_process', lambda argv, **o: (PHYSICAL, b'2.1.321', b''))
    with pytest.raises(ValueError, match='2.1.322'):
        vendor_guidance.read('cli://claude', str(tmp_path/'claude.exe'))
    assert normalized('cc_version=2.1.321.b46;') == 'cc_version=<CLI version>;'


def test_transport_texts_are_what_the_codecs_build():
    texts = {t['id']: t for t in transport_texts()}
    assert texts['claude-native']['text'] == native_cli.NATIVE_INSTRUCTIONS
    assert texts['claude-toolless']['text'] == native_cli.INSTRUCTIONS
    assert texts['agy-action-guidance']['text'] == native_cli.AGY_ACTION_GUIDANCE
    assert texts['codex-structured']['text'].startswith(
        'Tare role guidance for choosing the external decision:\n<role guidance>')
    assert all(t['endpoint'] in native_cli.CLI_MINIMUM and t['title'] and t['text'] for t in texts.values())


def test_agy_offers_nothing_to_choose(tmp_path):
    with pytest.raises(ValueError):
        native_cli.VendorCliClient('cli://agy', executable=str(tmp_path/'agy.exe'), version='1.2.7',
                                   vendor_guidance={'environment': None})


@pytest.mark.parametrize('codec', ['claude-native', 'claude-plain', 'claude-summary', 'claude-toolless',
                                   'claude-structured'])
def test_claude_choices_travel_as_a_mod_with_hooks_on_and_nothing_else_changes(tmp_path, monkeypatch, codec):
    choices = {'context:userEmail': None, 'attachment:environment': None,
               'attachment:date': 'Today is a working day.'}
    seen = dispatched(tmp_path, monkeypatch, codec, version=CLAUDE_MEASURED, switches=WITH_MOD,
                      vendor_guidance=choices)
    argv, files = list(seen['argv']), seen['files']
    at = argv.index('--plugin-dir')
    assert Path(argv[at + 1]) == seen['root']/'tare-guidance'
    source = files.pop('tare-guidance/hooks/register.ts')
    assert 'const dropContext: string[] = ["userEmail"];' in source
    assert ('const attachments: Record<string, string | null> = '
            '{"date": "Today is a working day.", "environment": null};') in source
    assert 'const reading: boolean = false;' in source  # a call marks nothing; only a reading does
    assert json.loads(files.pop('tare-guidance/hooks/hooks.json')) == {'modules': ['./register.ts']}
    assert json.loads(files.pop('tare-guidance/.claude-plugin/plugin.json'))['name'] == 'tare-guidance'
    settings = json.loads(argv[argv.index('--settings') + 1])
    assert 'disableAllHooks' not in settings and settings['enabledPlugins'] == {}  # the mod's hooks can run
    turned_on = codec != 'claude-native'  # the native codec with tools already runs with hooks on
    assert seen['receipt']['vendor_guidance'] == {
        'choices': {'attachment:date': 'replace', 'attachment:environment': 'remove',
                    'context:userEmail': 'remove'}, 'sha256': sha(choices),
        'hooks_turned_on': turned_on}
    # Only the mod and, where tare keeps hooks off, that switch differ from a call without a choice.
    if turned_on:
        argv[argv.index('--settings') + 1] = native_cli.encode({**settings, 'disableAllHooks': True})
    assert fingerprint({**seen, 'argv': argv[:at] + argv[at + 2:]}) == BEFORE[codec]


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
                                     qualification=AnyQualified(), vendor_guidance={'attachment:date': None})
    _, receipt = cli.post_observed('', native_cli.cli_request(json.loads(json.dumps(CHAT)), native=True), execution())
    assert receipt['vendor_guidance']['mod_listed_at_init'] is True


@pytest.mark.parametrize('version,switches,code', [('2.1.321', WITH_MOD, 'CLI_GUIDANCE_UNSUPPORTED'),
                                                   ('2.1.322', native_cli.CLI_SWITCHES['cli://claude'],
                                                    'CLI_CAPABILITY_UNAVAILABLE')])
def test_a_claude_that_cannot_load_the_mod_is_refused_before_dispatch(tmp_path, monkeypatch, version, switches, code):
    seen = dispatched(tmp_path, monkeypatch, 'claude-native', version=version, switches=switches,
                      vendor_guidance={'attachment:date': None}, expect=ModelFailure)
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


def test_every_placeholder_of_a_claude_block_has_a_pattern_that_finds_its_value():
    """An edited block is sent with the values the CLI wrote, so each placeholder `normalized` writes into a Claude
    attachment needs a FILL pattern that finds that value in the CLI's own text."""
    import re
    from compute_plane.vendor_guidance import FILL, normalized
    own = ('# Environment\n - Primary working directory: C:/Temp/tare-reading-x\n - OS Version: Windows 11 10.0.1\n'
           'The user\'s email address is someone@example.com.\n'
           'You are powered by the model named Sonnet 5.5. The exact model ID is claude-sonnet-5-5. '
           'Assistant knowledge cutoff is June 2026.\n\nToday\'s date is 2026-11-12.\n'
           '<total_tokens>15000000 tokens left</total_tokens>')
    shown = normalized(own, ['C:/Temp/tare-reading-x'])
    values = {name: re.search(pattern, own)[1] for name, pattern in FILL}
    assert set(re.findall(r'<[a-zA-Z ]+>', shown)) == set(values)
    assert values == {'<date>': '2026-11-12', '<OS version>': 'Windows 11 10.0.1',
                      '<working directory>': 'C:/Temp/tare-reading-x', '<model name>': 'Sonnet 5.5',
                      '<model id>': 'claude-sonnet-5-5', '<knowledge cutoff>': 'June 2026',
                      '<account email>': 'someone@example.com', '<tokens>': '15000000'}
    for name, value in values.items():  # what the mod does, placeholder by placeholder, gives the CLI's text back
        shown = shown.replace(name, value)
    assert shown == own


def test_a_real_claude_reading_records_each_piece_and_says_which_the_local_reading_missed(tmp_path, monkeypatch):
    import urllib.error
    import urllib.request
    real = []

    def process(argv, **options):
        if argv[-1] == '--version':
            return PHYSICAL, b'2.1.322 (Claude Code)', b''
        root = options['cwd']
        source = (root/'tare-guidance/hooks/register.ts').read_text(encoding='utf-8')
        if 'ANTHROPIC_BASE_URL' in options['env']:  # the zero-token reading first
            request = urllib.request.Request(options['env']['ANTHROPIC_BASE_URL'] + '/v1/messages',
                                             data=json.dumps(recorded_claude(root)).encode(), method='POST')
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(request, timeout=10)
            return PHYSICAL | {'protocol_complete': True}, b'', b''
        real.append(argv)  # the real call: tare's mod records every piece as sent, one file each
        assert 'const recording: boolean = true;' in source and 'const reading: boolean = false;' in source
        assert 'disableAllHooks' not in json.loads(argv[argv.index('--settings') + 1])
        email = "The user's email address is someone@example.com."
        reminder = '<total_tokens>15000000 tokens left</total_tokens>'
        pieces = [{'kind': 'context', 'blocks': [{'name': 'userEmail', 'text': email},
                                                 {'name': 'currentDate', 'text': "Today's date is 2026-11-12."}]},
                  {'kind': 'attachment', 'type': 'session_context',
                   'text': 'As you answer, use this context:\n# userEmail\n' + email + '\n\nAttached by Claude Code.'},
                  {'kind': 'attachment', 'type': 'date', 'text': "Today's date is 2026-11-12."},
                  {'kind': 'attachment', 'type': 'total_tokens_reminder',
                   'text': None if '"total_tokens_reminder": null' in source else reminder}]
        (root/'guidance-record').mkdir()
        for number, piece in enumerate(pieces, 1):
            (root/'guidance-record'/f'{number}.json').write_text(json.dumps(piece), encoding='utf-8')
        result = {'type': 'result', 'subtype': 'success', 'is_error': False, 'usage': {'output_tokens': 4},
                  'total_cost_usd': 0.001}
        return PHYSICAL, json.dumps(result).encode(), b''
    monkeypatch.setattr(vendor_guidance, 'run_pipe_process', process)
    value = vendor_guidance.read_real('cli://claude', str(tmp_path/'claude.exe'), model='claude-haiku-5-5')
    items = {item['id']: item for item in value['items']}
    # currentDate is sent as the engine's own date attachment, not as a block
    assert list(items) == ['context:userEmail', 'attachment:session_context', 'attachment:date',
                           'attachment:total_tokens_reminder']
    assert {k: i['local'] for k, i in items.items()} == {
        'context:userEmail': 'same', 'attachment:session_context': 'same', 'attachment:date': 'same',
        'attachment:total_tokens_reminder': 'absent'}
    assert items['attachment:total_tokens_reminder']['text'] == '<total_tokens><tokens> tokens left</total_tokens>'
    assert value['usage'] == {'output_tokens': 4, 'total_cost_usd': 0.001} and len(real) == 1
    removed = vendor_guidance.read_real('cli://claude', str(tmp_path/'claude.exe'), model='claude-haiku-5-5',
                                        choices={'attachment:total_tokens_reminder': None})
    assert removed['left_out'] == ['attachment:total_tokens_reminder']
    assert 'attachment:total_tokens_reminder' not in {i['id'] for i in removed['items']}
    with pytest.raises(ValueError, match='no real reading'):
        vendor_guidance.read_real('cli://codex', 'codex.exe', model='gpt-6-luna')


def protobuf(number, value):
    """One length-delimited protobuf field, as AGY's record holds its texts."""
    data = value.encode('utf-8') if isinstance(value, str) else value
    out = bytearray()
    for part in ((number << 3) | 2, len(data)):
        while part > 0x7F:
            out.append(part & 0x7F | 0x80)
            part >>= 7
        out.append(part)
    return bytes(out) + data


def agy_tool(name, text, schema):
    return protobuf(8, protobuf(1, name) + protobuf(2, text) + protobuf(3, schema))


def agy_record(system, prompt, *, marks=True):
    """An AGY 1.5.7 generation record: tare's system text and request in AGY's frames, its blocks and tools."""
    framed = '<System Prompt>\r\n' + system + '\r\n</System Prompt>' if marks else system
    request = ('<USER_REQUEST>\n' + prompt + '\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nThe current local time is: '
               '2026-11-12T13:37:07-03:00.\n</ADDITIONAL_METADATA>\n<USER_SETTINGS_CHANGE>\nThe user changed setting '
               '`Model Selection` from None to Gemini 3.8 Flash (Low). No need to comment.\n</USER_SETTINGS_CHANGE>')
    return protobuf(1, protobuf(1, framed) + protobuf(2, protobuf(3, request))
                    + agy_tool('finish', 'End your task.', '{"type":"object"}')
                    + agy_tool('manage_task', 'Manage background tasks.', '{"properties":{}}'))


def test_an_agy_reading_finds_what_agy_added_by_its_marks_and_refuses_to_guess():
    system = 'Line one of tare.\nLine two.'
    items = {item['id']: item for item in vendor_guidance.agy_items(agy_record(system, '{"x":1}'), system)}
    assert list(items) == ['system:frame', 'request:frame', 'request:ADDITIONAL_METADATA',
                           'request:USER_SETTINGS_CHANGE', 'tool:finish', 'tool:manage_task']
    assert items['system:frame']['text'] == "<System Prompt>\n<tare's system text>\n</System Prompt>"
    assert items['request:ADDITIONAL_METADATA']['text'] == 'The current local time is: <local time>.'
    assert 'from None to <model name>. No need' in items['request:USER_SETTINGS_CHANGE']['text']
    assert items['tool:finish']['text'] == 'End your task.'  # its schema is tare's own
    assert items['tool:manage_task']['text'] == 'Manage background tasks.\n\nParameters: {"properties":{}}'
    assert all(item['controls'] == () for item in items.values())
    with pytest.raises(ValueError, match='could not be read'):
        vendor_guidance.agy_items(agy_record(system, '{}', marks=False), system)
    assert [i['id'] for i in vendor_guidance.AGY_STARTING_POINT['items']] == list(items)


def test_a_real_agy_reading_opens_only_the_conversation_its_call_reported(tmp_path, monkeypatch):
    import sqlite3
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    store = tmp_path/'.gemini'/'antigravity-cli'/'conversations'
    store.mkdir(parents=True)
    ours = ['0f0e0d0c-0b0a-0908-0706-050403020100']

    def process(argv, **options):
        if argv[-1] == '--version':
            return PHYSICAL, b'1.5.7', b''
        agent = (options['cwd']/'.agents/agents/tare-inference/agent.md').read_text(encoding='utf-8')
        if ours[0].startswith('0'):
            database = sqlite3.connect(store/(ours[0] + '.db'))
            database.execute('create table gen_metadata (idx integer, data blob)')
            database.execute('insert into gen_metadata values (0, ?)', (agy_record(agent.split('---\n')[-1], '{}'),))
            database.commit()
            database.close()
        rows = [{'event': 'init', 'conversation_id': ours[0], 'init': {}},
                {'event': 'result', 'result': {'status': 'SUCCESS', 'usage': {'total_tokens': 758}}}]
        return PHYSICAL, '\n'.join(json.dumps(r) for r in rows).encode(), b''
    monkeypatch.setattr(vendor_guidance, 'run_pipe_process', process)
    value = vendor_guidance.read_real('cli://agy', str(tmp_path/'agy.exe'), model='gemini-3.8-flash-low')
    assert value['cli_version'] == '1.5.7' and value['usage'] == {'total_tokens': 758}
    assert 'tool:manage_task' in {i['id'] for i in value['items']}
    ours[0] = '1f1e1d1c-1b1a-1918-1716-151413121110'  # the call reports a conversation AGY never wrote
    with pytest.raises(ValueError, match='no record of its own'):
        vendor_guidance.read_real('cli://agy', str(tmp_path/'agy.exe'), model='gemini-3.8-flash-low')


def test_the_codex_reading_lists_the_installed_skills_as_exec_does_without_login_or_config(tmp_path, monkeypatch):
    from compute_plane.native_cli_codex import prompt_input
    home = tmp_path/'codex-home'
    skill = home/'skills'/'linked-project'
    (skill/'agents').mkdir(parents=True)
    (skill/'.claude-plugin').mkdir()
    (skill/'SKILL.md').write_text('---\nname: x\n---\n', encoding='utf-8')
    (skill/'agents/openai.yaml').write_text('policy:\n  allow_implicit_invocation: false\n', encoding='utf-8')
    (skill/'.claude-plugin/plugin.json').write_text('{"name": "linked"}', encoding='utf-8')
    (skill/'big-project-file.bin').write_bytes(b'0' * 1000)
    (home/'config.toml').write_text('model = "other"\n', encoding='utf-8')
    (home/'auth.json').write_text('{}', encoding='utf-8')
    monkeypatch.setenv('CODEX_HOME', str(home))
    seen = []

    def process(argv, **options):
        copy = Path(options['env']['CODEX_HOME'])
        seen.append(sorted(p.relative_to(copy).as_posix() for p in copy.rglob('*') if p.is_file()))
        assert 'approval_policy="never"' in argv and 'sandbox_mode="read-only"' in argv
        return PHYSICAL, b'[]', b''
    body = native_cli.cli_request({'model': 'fixture', 'messages': []}, structured=True, native_arguments=True)
    (tmp_path/'root').mkdir()
    prompt_input('codex.exe', tmp_path/'root', body, {'slug': 'fixture'}, (), {}, execution(), process, {})
    assert [p for p in seen[0] if p.startswith('skills/')] == [
        'skills/linked-project/.claude-plugin/plugin.json', 'skills/linked-project/SKILL.md',
        'skills/linked-project/agents/openai.yaml']  # no project file
    assert 'config.toml' not in seen[0] and 'auth.json' not in seen[0]  # no login, no owner's configuration
