"""Optional, pinned CPU Von observations; no downloads or routing authority."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from .choice_assessment import validate_answer, validate_request
from .laya_choice_worker import file_hash

MODEL_REVISION = 'aa2fdc9630ecdadef32c56073553b3a69bed38bf'
SOURCE_REVISION = '2656a69be2ef0ebf2f3058302e52791f99cbd8de'
FILES = {'option_marker.pt', 'config.json', 'tokenizer.json',
         'tokenizer_config.json', 'marker_calibration.json'}
MAX_TOKENS = 2048


def pack_request(model, request):
    """Reject special-token injection and overflow before the forward pass."""
    validate_request(request)
    values = [request['state'], request['question'], *request['options'].values()]
    if any(token in text for token in model.tokenizer.all_special_tokens for text in values):
        raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
    packed = model.pack_sequence(request['state'], request['question'],
                                 list(request['options'].values()))
    inputs = model.tokenizer(packed, return_tensors='pt', truncation=False)
    tokens = inputs['input_ids'][0]
    if len(tokens) > MAX_TOKENS:
        raise ValueError('ASSESSMENT_CAPACITY_EXCEEDED')
    positions = (tokens == model.mask_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != len(request['options']):
        raise ValueError('ASSESSMENT_REQUEST_INVALID')
    return inputs, positions


class VonChoiceModel:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if (manifest.get('schema') != 'tare.local-labs/von-checkpoint/1'
                or manifest.get('model') != 'wfzyx/von-1.0'
                or manifest.get('revision') != MODEL_REVISION
                or manifest.get('source_revision') != SOURCE_REVISION
                or set(manifest.get('files', {})) != FILES):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        root = path.parent / manifest['directory']
        for name in FILES:
            if file_hash(root / name) != manifest['files'][name]:
                raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          CUDA_VISIBLE_DEVICES='')
        import torch
        import transformers
        from ._vendor.von.option_marker import OptionMarkerModel, OptionMarkerScorer
        # The measured compatibility envelope, not an untested open-ended SDK range.
        if torch.__version__ != '2.8.0+cpu' or transformers.__version__ != '4.57.6':
            raise ValueError('ASSESSMENT_ENVIRONMENT_CHANGED')
        torch.set_num_threads(2)
        config_data = json.loads((root / 'config.json').read_bytes())
        if (config_data.get('model_type') != 'modernbert'
                or config_data.get('max_position_embeddings') != MAX_TOKENS):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        # Translate the new config's RoPE spelling to this pinned Transformers
        # version explicitly, preserving both trained positional frequencies.
        rope = config_data['rope_parameters']
        if any(rope[k]['rope_type'] != 'default' for k in ('full_attention', 'sliding_attention')):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        config = transformers.AutoConfig.from_pretrained(root, local_files_only=True,
                                                         trust_remote_code=False)
        config.global_rope_theta = rope['full_attention']['rope_theta']
        config.local_rope_theta = rope['sliding_attention']['rope_theta']
        config.reference_compile = False

        class LocalModel(OptionMarkerModel):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.encoder = transformers.AutoModel.from_config(config, attn_implementation='sdpa')
                self.hidden_size = config.hidden_size
                self.scorer = OptionMarkerScorer(hidden_size=self.hidden_size)
                # The artifact names a newer TokenizersBackend; its serialized
                # tokenizer is loaded directly without inference-time conversion.
                tokens = json.loads((root / 'tokenizer_config.json').read_bytes())
                special = {k: tokens[k] for k in ('cls_token', 'mask_token', 'pad_token',
                                                  'sep_token', 'unk_token')}
                self.tokenizer = transformers.PreTrainedTokenizerFast(
                    tokenizer_file=str(root / 'tokenizer.json'),
                    model_max_length=MAX_TOKENS, **special)
                self.mask_token_id = self.tokenizer.mask_token_id

        self.model = LocalModel()
        state = torch.load(root / 'option_marker.pt', map_location='cpu',
                           weights_only=True, mmap=True)
        self.model.load_state_dict(state, strict=True, assign=True)
        self.model.eval()
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        self.temperature = json.loads((root / 'marker_calibration.json').read_bytes())['temperature']
        if (type(self.temperature) not in (int, float) or not math.isfinite(self.temperature)
                or self.temperature <= 0):
            raise ValueError('ASSESSMENT_CHECKPOINT_CHANGED')
        self.identity = {'model': manifest['model'], 'revision': MODEL_REVISION,
                         'source_revision': SOURCE_REVISION, 'device': 'cpu', 'threads': 2,
                         'method': 'von-option-marker-cpu-v1', 'temperature': self.temperature,
                         'max_tokens': MAX_TOKENS, 'torch': torch.__version__,
                         'transformers': transformers.__version__,
                         'manifest_sha256': hashlib.sha256(raw).hexdigest()}

    def assess(self, request):
        import torch
        inputs, positions = pack_request(self.model, request)
        with torch.inference_mode():
            logits = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                mask_positions=[positions])[0]
            values = torch.softmax(logits / self.temperature, dim=-1).tolist()
        probabilities = dict(zip(request['options'], values))
        return validate_answer({'choice': max(probabilities, key=probabilities.get),
                                'probabilities': probabilities,
                                'usage': {'input_tokens': len(inputs['input_ids'][0]), 'output_tokens': 0},
                                'model_identity': self.identity}, request)


def main():
    from .choice_assessment import WORKER_FAILURES
    try:
        raw = sys.stdin.buffer.read(8193)
        if len(raw) > 8192:
            raise ValueError('ASSESSMENT_INPUT_OVERFLOW')
        request = json.loads(raw)
        validate_request(request)
        with contextlib.redirect_stdout(sys.stderr):
            result = VonChoiceModel(sys.argv[1]).assess(request)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        result = {'failure': str(exc) if str(exc) in WORKER_FAILURES else 'ASSESSMENT_WORKER_FAILED'}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
