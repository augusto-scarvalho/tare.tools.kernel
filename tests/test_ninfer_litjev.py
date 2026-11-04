import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
import urllib.error

import pytest

from compute_plane import ninfer_litjev

REQUEST = {'state': 'Rename one variable.', 'question': 'How hard is this task?',
           'options': {'easy': 'Mechanical change.', 'hard': 'Needs design work.', 'unclear': 'Not enough detail.'}}


@pytest.fixture
def ninfer():
    seen = {'status': 200, 'bodies': [], 'generations': 0,
            'top': [('A', -0.1), (' A', -2.5), ('B', -2.4), ('<|im_end|>', -8.0)]}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen['bodies'].append(seen['body'])
            if seen.get('legacy') and 'logprobs' in seen['body']:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({'error': {'type': 'invalid_request_error',
                    'code': 'logprobs_not_supported', 'param': 'logprobs'}}).encode())
                return
            if seen['status'] != 200:
                self.send_response(seen['status'])
                self.end_headers()
                self.wfile.write(seen.get('error', b'{}'))
                return
            seen['generations'] += 1
            top = [{'token': t, 'logprob': p, 'bytes': list(t.encode())} for t, p in seen['top']]
            reply = {'model': 'qwen38-gsq', 'usage': {'prompt_tokens': 120, 'completion_tokens': 1},
                     'choices': [{'message': {'content': 'A'}, 'logprobs': {'content': [
                         {'token': 'A', 'logprob': -0.1, 'bytes': [65], 'top_logprobs': top}]}}]}
            self.send_response(seen['status'])
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen['url'] = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    yield seen
    server.shutdown()


def test_first_token_alternatives_become_a_restricted_softmax(ninfer):
    result = ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='qwen38-gsq')
    body = ninfer['body']
    assert body['logprobs'] is True and body['top_logprobs'] == 20 and body['max_tokens'] == 1
    assert len(ninfer['bodies']) == ninfer['generations'] == 1
    assert body['chat_template_kwargs'] == {'enable_thinking': False} and 'A' in body['messages'][1]['content']
    a = math.exp(-0.1) + math.exp(-2.5)  # 'A' and ' A' are one answer
    assert result['probabilities']['easy'] == pytest.approx(a / (a + math.exp(-2.4)))
    assert result['probabilities']['unclear'] == 0.0  # outside the top 20
    assert result['choice'] == 'easy' and result['usage'] == {'input_tokens': 120, 'output_tokens': 1}
    assert result['model_identity']['method'] == ninfer_litjev.METHOD


def test_no_option_code_or_a_refused_request_is_a_named_failure(ninfer):
    ninfer['top'] = [('The', -0.2), ('<|im_end|>', -3.0)]
    with pytest.raises(ValueError, match='MISSING_CANDIDATE'):
        ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='m')
    ninfer['status'] = 400  # a server started without --first-token-logprobs
    with pytest.raises(ValueError, match='BACKEND_UNAVAILABLE'):
        ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='m')


def test_legacy_validation_refusal_allows_one_generation_without_logprobs(ninfer):
    ninfer['legacy'] = True
    result = ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='qwen38-gsq')
    assert result['choice'] == 'easy' and ninfer['generations'] == 1
    modern, legacy = ninfer['bodies']
    assert modern['logprobs'] is True and 'logprobs' not in legacy
    assert {k: v for k, v in modern.items() if k != 'logprobs'} == legacy


@pytest.mark.parametrize('status,error', [
    (400, b'not JSON'),
    (400, b'{"error":{"type":"invalid_request_error","code":"logprobs_not_supported","param":"top_logprobs"}}'),
    (400, b'{"error":{"type":"invalid_request_error","code":"different_error","param":"logprobs"}}'),
    (400, b'{"error":{"type":"invalid_request_error","code":"logprobs_not_supported","param":"logprobs"},"padding":"' + b'x'*4096 + b'"}'),
    (500, b'{"error":{"type":"invalid_request_error","code":"logprobs_not_supported","param":"logprobs"}}'),
])
def test_other_or_unbounded_errors_never_repeat_a_request(ninfer, status, error):
    ninfer.update(status=status, error=error)
    with pytest.raises(ValueError, match='ASSESSMENT_'):
        ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='m')
    assert len(ninfer['bodies']) == 1 and ninfer['generations'] == 0


def test_legacy_retry_does_not_restart_the_timeout(ninfer, monkeypatch):
    ninfer['legacy'] = True
    ticks = iter((10, 10, 21))
    monkeypatch.setattr(ninfer_litjev, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(ValueError, match='BACKEND_UNAVAILABLE'):
        ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='m', timeout=10)
    assert len(ninfer['bodies']) == 1 and ninfer['generations'] == 0


def test_a_second_legacy_refusal_stops_without_a_third_request(ninfer):
    ninfer.update(legacy=True, status=400,
        error=b'{"error":{"type":"invalid_request_error","code":"logprobs_not_supported","param":"logprobs"}}')
    with pytest.raises(ValueError, match='BACKEND_UNAVAILABLE'):
        ninfer_litjev.assess(REQUEST, url=ninfer['url'], model='m')
    assert len(ninfer['bodies']) == 2 and ninfer['generations'] == 0


@pytest.mark.parametrize('error', [TimeoutError(), urllib.error.URLError('uncertain transport')])
def test_uncertain_transport_is_not_retried(monkeypatch, error):
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise error
    monkeypatch.setattr(ninfer_litjev.urllib.request, 'urlopen', fail)
    with pytest.raises(ValueError, match='BACKEND_UNAVAILABLE'):
        ninfer_litjev.assess(REQUEST, url='http://127.0.0.1:1', model='m')
    assert len(calls) == 1
