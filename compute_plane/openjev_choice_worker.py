"""Optional pinned OpenJEV NLI scorer. The CLI is CPU-only; no downloads."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from .choice_assessment import validate_request, validate_answer

FILES = {'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'}
LABELS = {'0': 'contradiction', '1': 'entailment', '2': 'neutral'}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint(manifest_path, *, include_weights=True):
    """Verify pinned artifacts; token-only admission may defer the weight hash.

    Model construction always uses the default full verification.
    """
    path = Path(manifest_path).resolve()
    value = json.loads(path.read_bytes())
    if (value.get('schema') != 'tare.local-labs/openjev-checkpoint/1'
            or set(value.get('files', {})) != FILES
            or not isinstance(value.get('directory'), str)):
        raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
    root = (path.parent / value['directory']).resolve()
    for name in FILES:
        if name == 'model.safetensors' and not include_weights:
            continue
        if sha256_file(root / name) != value['files'][name]:
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
    config = json.loads((root / 'config.json').read_bytes())
    if (config.get('architectures') != ['Qwen3_5ForSequenceClassification']
            or config.get('id2label') != LABELS
            or config.get('nli_template') != 'Premise: {premise}\nHypothesis: {hypothesis}'):
        raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
    return root, value, hashlib.sha256(path.read_bytes()).hexdigest()


def assessment_pairs(request):
    validate_request(request)
    premise = request['state'] + '\nQuestion: ' + request['question'] + '\nOptions:\n' + '\n'.join(
        key + ': ' + value for key, value in request['options'].items())
    return [(premise, 'The correct answer is: ' + option) for option in request['options'].values()]


def preflight_cpu_tokens(manifest_path, request, max_total_tokens):
    """Bound aggregate sequential CPU work before loading model weights.

    This operational cap is not a quality threshold or a latency guarantee.
    Token counting preserves every pair and uses the pinned local tokenizer.
    """
    if type(max_total_tokens) is not int or not 1 <= max_total_tokens <= 24576:
        raise ValueError('ASSESSMENT_EXECUTION_CONFIG_INVALID')
    pairs = assessment_pairs(request)
    root, manifest, _ = checkpoint(manifest_path, include_weights=False)
    os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    import transformers
    from transformers import AutoTokenizer
    if transformers.__version__ != manifest['transformers_version']:
        raise ValueError('ASSESSMENT_ENVIRONMENT_CHANGED')
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    template = json.loads((root / 'config.json').read_bytes())['nli_template']
    total = 0
    for premise, hypothesis in pairs:
        encoded = tokenizer(template.format(premise=premise.strip(), hypothesis=hypothesis.strip()),
                            truncation=False)
        count = len(encoded['input_ids'])
        if not 1 <= count <= 4096:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        total += count
    if total > max_total_tokens:
        raise ValueError('ASSESSMENT_CAPACITY_EXCEEDED')
    return total


def distribution(logits):
    if (len(logits) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in logits)):
        raise ValueError('ASSESSMENT_DISTRIBUTION_INVALID')
    peak = max(logits)
    shifted = [math.exp(x - peak) for x in logits]
    total = sum(shifted)
    return [x / total for x in shifted], logits[1] - peak - math.log(total)


def choice_answer(request, observations, identity):
    if len(observations) != len(request['options']):
        raise ValueError('ASSESSMENT_DISTRIBUTION_INVALID')
    # Normalize entailment scores across hypotheses, preserving the upstream
    # ranking. These are relative preferences, not calibrated Work success.
    scores = [distribution(row['logits'])[1] for row in observations]
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    total = sum(weights)
    probabilities = dict(zip(request['options'], (w / total for w in weights)))
    return validate_answer({'choice': max(probabilities, key=probabilities.get),
        'probabilities': probabilities,
        'usage': {'input_tokens': sum(row['input_tokens'] for row in observations), 'output_tokens': 0},
        'model_identity': identity}, request)


class OpenJevChoiceModel:
    def __init__(self, manifest_path, *, device='cpu', threads=4, cuda_layers=0):
        """GPU placements are probe-only and require external lease supervision.

        No automatic placement, quantization, remote code, fallback or routing.
        Production CLI below exposes CPU only. Local Labs owns GPU admission.
        """
        if (device not in {'cpu', 'cuda', 'hybrid'} or type(threads) is not int
                or not 1 <= threads <= 20 or (device == 'hybrid' and cuda_layers != 4)):
            raise ValueError('ASSESSMENT_PLACEMENT_INVALID')
        root, manifest, manifest_hash = checkpoint(manifest_path)
        os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if transformers.__version__ != manifest['transformers_version'] or torch.__version__ != manifest['torch_version']:
            raise ValueError('ASSESSMENT_ENVIRONMENT_CHANGED')
        torch.set_num_threads(threads)
        self.torch = torch
        # Accelerate may replace CPU-offloaded weights with meta placeholders.
        # Inputs go to a real device; its hooks materialize the offloaded weights.
        self.input_device = 'cuda:0' if device == 'cuda' else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.max_tokens = 4096
        dtype = torch.float32 if device == 'cpu' else torch.bfloat16
        kwargs = {}
        if device == 'cuda':
            kwargs['device_map'] = {'': 'cuda:0'}
        elif device == 'hybrid':
            kwargs['device_map'] = {'model.visual': 'cpu', 'model.language_model.embed_tokens': 'cpu',
                'model.language_model.rotary_emb': 'cpu', 'model.language_model.norm': 'cuda:0', 'score': 'cuda:0',
                **{f'model.language_model.layers.{n}': 'cuda:0' if n >= 28 else 'cpu' for n in range(32)}}
        self.model = AutoModelForSequenceClassification.from_pretrained(root, dtype=dtype,
            local_files_only=True, trust_remote_code=False, attn_implementation='eager', **kwargs).eval()
        if {str(k): v for k, v in self.model.config.id2label.items()} != LABELS:
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        self.identity = {k: manifest[k] for k in ('model', 'revision', 'subfolder', 'transformers_version', 'torch_version')}
        self.identity.update(device=device, dtype=str(dtype), threads=threads, cuda_layers=cuda_layers,
            manifest_sha256=manifest_hash, score_semantics='normalized-entailment-not-calibrated-success',
            max_pair_tokens=self.max_tokens, batching='sequential-pairs-no-prefix-cache',
            hybrid_semantics='cpu-weight-offload-gpu-execution' if device == 'hybrid' else None)
        self.last_observations = []

    def encode(self, premise, hypothesis):
        text = self.model.config.nli_template.format(premise=premise.strip(), hypothesis=hypothesis.strip())
        encoded = self.tokenizer(text, truncation=False, return_tensors='pt')
        if not 1 <= encoded['input_ids'].shape[1] <= self.max_tokens:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        return encoded

    def predict_encoded(self, encoded):
        with self.torch.inference_mode():
            values = {k: v.to(self.input_device) for k, v in encoded.items()}
            logits = self.model(**values, use_cache=False).logits[0].float().cpu().tolist()
        probabilities, _ = distribution(logits)
        return {'logits': logits, 'nli_probabilities': dict(zip(LABELS.values(), probabilities)),
                'input_tokens': encoded['input_ids'].shape[1], 'output_tokens': 0}

    def predict(self, premise, hypothesis):
        return self.predict_encoded(self.encode(premise, hypothesis))

    def assess(self, request):
        # Preflight ALL hypotheses before the first forward, no partial truncation.
        encoded = [self.encode(premise, hypothesis) for premise, hypothesis in assessment_pairs(request)]
        self.last_observations = [self.predict_encoded(value) for value in encoded]
        return choice_answer(request, self.last_observations, self.identity)


def main():
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        request = json.loads(raw)
        validate_request(request)
        with contextlib.redirect_stdout(sys.stderr):
            # FP32 CPU weights alone occupy ~17 GiB. An optional evaluator must
            # refuse a constrained host rather than displace the user's work.
            import psutil
            if psutil.virtual_memory().available < 36 * 1024**3:
                raise ValueError('ASSESSMENT_INSUFFICIENT_RAM')
            answer = OpenJevChoiceModel(sys.argv[1]).assess(request)
    except (ValueError, OSError, KeyError, ImportError, RuntimeError) as exc:
        reason = str(exc)
        answer = {'failure': reason if reason in {'ASSESSMENT_INPUT_OVERFLOW', 'ASSESSMENT_CHECKPOINT_CHANGED',
                                                 'ASSESSMENT_INSUFFICIENT_RAM'}
                  else 'ASSESSMENT_WORKER_FAILED'}
    print(json.dumps(answer, allow_nan=False))


if __name__ == '__main__':
    main()
