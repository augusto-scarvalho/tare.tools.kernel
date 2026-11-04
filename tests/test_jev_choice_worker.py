import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from compute_plane import jev_choice_worker as worker
from compute_plane.choice_assessment import run_assessment

REQUEST = {'state': 'Rename one variable.', 'question': 'How hard is this task?',
           'options': {'easy': 'Mechanical change.', 'hard': 'Needs design work.'}}


@pytest.fixture
def jev():
    seen = {'status': 200, 'reply': {'model': 'jev-1.13.0', 'usage': {'input_tokens': 90, 'output_tokens': 20},
                                     'answers': {'q': {'type': 'choice', 'choice': 'easy', 'confidence': 0.8,
                                                       'probabilities': {'easy': 0.9, 'hard': 0.1004}}}}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen['path'], seen['auth'] = self.path, self.headers['Authorization']
            seen['body'] = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            self.send_response(seen['status'])
            self.end_headers()
            self.wfile.write(json.dumps(seen['reply']).encode())

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    seen['url'] = f'http://127.0.0.1:{server.server_port}'
    yield seen
    server.shutdown()


def test_one_choice_question_maps_to_a_kernel_answer(jev):
    result = worker.assess(REQUEST, key='k', base_url=jev['url'])
    assert jev['path'] == '/v1/systemone' and jev['auth'] == 'Bearer k'
    assert jev['body'] == {'model': 'jev-1.13.0', 'state': REQUEST['state'], 'questions': {'q': {
        'type': 'choice', 'instructions': REQUEST['question'], 'criteria': REQUEST['options']}}}
    assert result['choice'] == 'easy' and sum(result['probabilities'].values()) == pytest.approx(1)
    assert result['usage'] == {'input_tokens': 90, 'output_tokens': 0}
    assert result['model_identity']['model'] == 'jev-1.13.0'


@pytest.mark.parametrize('status,reason', [(429, 'RESOURCE_BUSY'), (529, 'RESOURCE_BUSY'),
                                           (401, 'BACKEND_UNAVAILABLE'), (422, 'WORKER_FAILED')])
def test_http_failures_are_named_and_not_retried(jev, status, reason):
    jev['status'] = status
    with pytest.raises(ValueError, match=reason):
        worker.assess(REQUEST, key='k', base_url=jev['url'])


def test_an_answer_for_other_options_is_refused(jev):
    jev['reply']['answers']['q']['probabilities'] = {'easy': 1.0}
    with pytest.raises(ValueError):
        worker.assess(REQUEST, key='k', base_url=jev['url'])


def test_the_worker_runs_behind_run_assessment_and_needs_a_key(jev, monkeypatch):
    root = str(Path(worker.__file__).parents[1])
    command = [sys.executable, '-m', 'compute_plane.jev_choice_worker']
    for name, value in {'PYTHONPATH': root, 'TYPESAFE_BASE_URL': jev['url'], 'TYPESAFE_API_KEY': 'k' + chr(10)}.items():
        monkeypatch.setenv(name, value)
    assert run_assessment(command, REQUEST)['answer']['choice'] == 'easy'
    monkeypatch.delenv('TYPESAFE_API_KEY')
    assert run_assessment(command, REQUEST)['reason'] == 'ASSESSMENT_BACKEND_UNAVAILABLE'


def test_many_yes_no_questions_share_one_state(jev):
    jev['reply'] = {'model': 'jev-1.13.0', 'usage': {'input_tokens': 340, 'output_tokens': 0},
                    'answers': {'c1': {'type': 'noul', 'noul': 0.97}, 'c2': {'type': 'noul', 'noul': 0.02}}}
    answers, tokens = worker.assess_many('[c1] a\n[c2] b', {'c1': 'Does [c1] help?', 'c2': 'Does [c2] help?'},
                                         key='k', base_url=jev['url'])
    assert answers == {'c1': 0.97, 'c2': 0.02} and tokens == 340
    assert jev['body'] == {'model': 'jev-1.13.0', 'state': '[c1] a\n[c2] b', 'questions': {
        'c1': {'type': 'noul', 'instructions': 'Does [c1] help?'},
        'c2': {'type': 'noul', 'instructions': 'Does [c2] help?'}}}


@pytest.mark.parametrize('state,questions', [('', {'q': 'Yes?'}), ('s', {}), ('s', {'q': ' '}), ('s', {'x'*33: 'Yes?'}),
                                             ('s', {f'q{i}': 'Yes?' for i in range(worker.MAX_QUESTIONS + 1)})])
def test_many_questions_refuse_malformed_requests_before_sending(jev, state, questions):
    jev['body'] = None
    with pytest.raises(ValueError, match='ASSESSMENT_REQUEST_INVALID'):
        worker.assess_many(state, questions, key='k', base_url=jev['url'])
    assert jev['body'] is None


@pytest.mark.parametrize('reply,reason', [({'answers': {}, 'usage': {'input_tokens': 1}}, 'INCOMPLETE_READOUT'),
                                          ({'answers': {'q': {'noul': 1.5}}, 'usage': {'input_tokens': 1}},
                                           'MODIFIED_READOUT')])
def test_many_questions_refuse_answers_they_did_not_ask_for(jev, reply, reason):
    jev['reply'] = {'model': worker.MODEL, **reply}
    with pytest.raises(ValueError, match=reason):
        worker.assess_many('s', {'q': 'Yes?'}, key='k', base_url=jev['url'])


@pytest.mark.parametrize('returned', ['jev-1.14.0', 'jev-1.13.0', None])
def test_batch_uses_only_the_requested_checkpoint(jev, returned):
    jev['reply']={'model':returned,'usage':{'input_tokens':42},
                  'answers':{'q':{'type':'noul','noul':0.95}}}
    if returned=='jev-1.14.0':
        assert worker.assess_many('evidence',{'q':'Supported?'},key='k',model='jev-1.14.0',
                                 base_url=jev['url'])==({'q':0.95},42)
    else:
        with pytest.raises(ValueError,match='ASSESSMENT_CHECKPOINT_CHANGED'):
            worker.assess_many('evidence',{'q':'Supported?'},key='k',model='jev-1.14.0',base_url=jev['url'])
    assert jev['body']['model']=='jev-1.14.0'


def test_a_state_over_the_model_window_is_an_input_overflow(jev):
    jev['status'], jev['reply'] = 400, {'detail': {'error_type': 'max_tokens_exceeded'}}
    with pytest.raises(ValueError, match='ASSESSMENT_INPUT_OVERFLOW'):
        worker.assess_many('s', {'q': 'Yes?'}, key='k', base_url=jev['url'])


def test_one_request_mixes_yes_no_and_choice_questions(jev):
    jev['reply'] = {'model': 'jev-1.13.0', 'usage': {'input_tokens': 410, 'output_tokens': 0}, 'answers': {
        'c1': {'type': 'noul', 'noul': 0.95},
        's': {'type': 'choice', 'choice': 'partial', 'probabilities': {'answered': 0.2, 'partial': 0.6, 'none': 0.2004}}}}
    answers, tokens = worker.assess_many('report', {'c1': 'Does [c1] show it?', 's': {
        'question': 'Does the report answer?', 'options': {'answered': 'Yes.', 'partial': 'Partly.', 'none': 'No.'}}},
        key='k', base_url=jev['url'])
    assert answers['c1'] == 0.95 and answers['s']['choice'] == 'partial' and tokens == 410
    assert sum(answers['s']['probabilities'].values()) == pytest.approx(1)
    assert jev['body']['questions']['s'] == {'type': 'choice', 'instructions': 'Does the report answer?',
                                             'criteria': {'answered': 'Yes.', 'partial': 'Partly.', 'none': 'No.'}}


@pytest.mark.parametrize('question', [{'question': 'Q?', 'options': {'only': 'One.'}}, {'question': 'Q?'},
                                      {'question': ' ', 'options': {'a': 'A', 'b': 'B'}}, 3])
def test_a_malformed_choice_among_many_questions_is_refused_before_sending(jev, question):
    jev['body'] = None
    with pytest.raises(ValueError, match='ASSESSMENT_REQUEST_INVALID'):
        worker.assess_many('s', {'c1': 'Yes?', 'x': question}, key='k', base_url=jev['url'])
    assert jev['body'] is None
