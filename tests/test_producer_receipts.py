"""Consumer distrusts claimed identity, stale subject, and incomplete artifact bytes."""
import copy
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from assurance_plane.producer_receipts import ProducerSigner, verify_receipt


@pytest.fixture
def signed():
    signer = ProducerSigner(Ed25519PrivateKey.generate())
    subject = dict(work_id='work', session_id='session', effect_id='effect', binding_id='binding',
                   fence_seq=1, run_id='run', **{name:'a'*64 for name in (
                       'action_sha256','source_sha256','environment_sha256','runtime_sha256','profile_sha256')})
    artifacts = {'stdout.txt': b'known output', 'stderr.txt': b'', 'process.json': b'{"exit_code":0}'}
    keys = {signer.producer_id: signer.public_key_bytes}
    return signer, subject, artifacts, keys


def test_admitted_producer_and_all_bytes_validate_without_aliasing(signed):
    signer, subject, artifacts, keys = signed
    receipt = signer.sign(subject, artifacts)
    result = verify_receipt(receipt, artifacts, subject, keys)
    assert result['subject'] == subject
    result['subject']['fence_seq'] = 100
    assert receipt['payload']['subject']['fence_seq'] == 1
    assert 'private_key' not in repr(receipt)


@pytest.mark.parametrize('field', ['work_id','session_id','effect_id','binding_id','fence_seq','run_id',
                                  'action_sha256','source_sha256','environment_sha256','runtime_sha256','profile_sha256'])
def test_stale_or_other_subject_is_refused(signed, field):
    signer, subject, artifacts, keys = signed
    receipt = signer.sign(subject, artifacts)
    stale = dict(subject); stale[field] = 2 if field == 'fence_seq' else ('b'*64 if field.endswith('_sha256') else 'other')
    with pytest.raises(ValueError, match='subject'):
        verify_receipt(receipt, artifacts, stale, keys)


@pytest.mark.parametrize('change', ['missing','extra','bytes','manifest','signature','issuer','self-key'])
def test_tampered_missing_extra_and_self_attested_result_fail(signed, change):
    signer, subject, artifacts, keys = signed
    receipt = signer.sign(subject, artifacts); supplied = dict(artifacts)
    if change == 'missing': supplied.pop('stderr.txt')
    elif change == 'extra': supplied['unlisted.txt'] = b'new'
    elif change == 'bytes': supplied['process.json'] = b'{"exit_code":1}'
    elif change == 'manifest': receipt['payload']['artifacts']['stdout.txt']['bytes'] = True
    elif change == 'signature': receipt['signature'] = 'A'*86+'=='
    elif change == 'issuer': keys = {}
    elif change == 'self-key': receipt['public_key'] = signer.public_key_bytes.hex()
    with pytest.raises(ValueError): verify_receipt(receipt, supplied, subject, keys)


@pytest.mark.parametrize('name', ['/absolute','../escape','x/../escape','x\\y','x:ads','NUL.txt','CON',
                                 'COM1.log','x/','x.','x ','unicodé','a\0b'])
def test_unsafe_artifact_names_cannot_be_signed(signed, name):
    signer, subject, _, _ = signed
    with pytest.raises(ValueError): signer.sign(subject, {name:b'x'})


def test_bounds_case_collision_and_unknown_signer_are_refused(signed):
    signer, subject, artifacts, keys = signed
    for value in ({'a':b'x','A':b'y'}, {str(i):b'' for i in range(65)}, {'big':b'x'*(8*1024**2+1)}):
        with pytest.raises(ValueError): signer.sign(subject, value)
    foreign = ProducerSigner(Ed25519PrivateKey.generate())
    with pytest.raises(ValueError, match='not admitted'):
        verify_receipt(foreign.sign(subject, artifacts), artifacts, subject, keys)
