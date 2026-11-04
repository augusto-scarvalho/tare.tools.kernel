"""Optional CPU-only Laya adapter. No download, GPU selection or workflow logic."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

from .choice_assessment import validate_request, validate_answer

FILES = ('model.safetensors', 'rl_agent_config.json', 'encoder/config.json',
         'tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json')


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def check_envelope(agent, request):
    """Reject every SDK truncation path, including individual option truncation."""
    validate_request(request)
    tok = agent.tok
    texts = [request['state'], request['question'], *request['options'], *request['options'].values()]
    if any(tok.mask_token in text for text in texts):
        raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
    def count(text):
        return len(tok(text, add_special_tokens=False)['input_ids'])
    options = [count(' ' + k + ': ' + v) for k, v in request['options'].items()]
    head = count('choice question: ' + request['question']) + sum(n + 1 for n in options)
    if (max(options) > 48 or head > agent.cfg.get('head_max_len', 192)
            or head + count(request['state']) + 4 > agent.cfg.get('max_len', 512)):
        raise ValueError('ASSESSMENT_INPUT_OVERFLOW')


class LayaChoiceModel:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_bytes())
        if (manifest.get('schema') != 'tare.local-labs/laya-checkpoint/1'
                or set(manifest.get('files', {})) != set(FILES)
                or manifest.get('sdk_version') != '0.3.4'):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        root = path.parent / manifest['directory']
        for name in FILES:
            actual = file_hash(root / name)
            if actual != manifest['files'][name]:
                raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        # Imports are optional; ordinary Kernel users do not depend on torch.
        os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        import torch
        import laya
        if laya.__version__ != manifest['sdk_version']:
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        torch.set_num_threads(2)
        self.agent = laya.load(str(root), device='cpu')
        if self.agent.device.type != 'cpu':
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        # Laya can rewrite tokenizer configuration at load; never hide that drift.
        for name in FILES[1:]:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest['files'][name]:
                raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        self.identity = {k: manifest[k] for k in ('model', 'revision', 'subfolder', 'sdk_version')}
        self.identity.update(device='cpu', threads=2, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def assess(self, request):
        check_envelope(self.agent, request)
        result = self.agent.predict(request['state'], {'demand': {
            'type': 'choice', 'instructions': request['question'], 'criteria': request['options']}})
        answer = result['answers']['demand']
        return validate_answer({**answer, 'usage': result['usage'], 'model_identity': self.identity}, request)


def main():
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        request = json.loads(raw)
        validate_request(request)
        with contextlib.redirect_stdout(sys.stderr):
            result = LayaChoiceModel(sys.argv[1]).assess(request)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        code = str(exc)
        result = {'failure': code if code in {'ASSESSMENT_INPUT_OVERFLOW', 'ASSESSMENT_CHECKPOINT_CHANGED'}
                  else 'ASSESSMENT_WORKER_FAILED'}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
