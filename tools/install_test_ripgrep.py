"""Install the pinned upstream test executable under RUNNER_TEMP, never globally."""
import hashlib
import io
import os
from pathlib import Path
import tarfile
import time
import urllib.error
import urllib.request
import zipfile


def fetch(url, attempts=5, sleep=time.sleep):
    """GitHub's release download answers 5xx or times out now and then (504s failed every shard on 2026-11-09);
    a shard waits that out. A 4xx other than 429 is not transient and fails at once, like a digest mismatch."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read(8*1024*1024+1)
        except OSError as error:  # URLError, HTTPError, a reset connection or a timeout
            code = getattr(error, 'code', None)
            if (code is not None and code != 429 and code < 500) or attempt == attempts-1:
                raise
            sleep(2**(attempt+1))


def main():
    asset, digest, executable = (
        ('ripgrep-14.1.1-x86_64-pc-windows-msvc.zip',
         'd0f534024c42afd6cb4d38907c25cd2b249b79bbe6cc1dbee8e3e37c2b6e25a1', 'rg.exe') if os.name == 'nt' else
        ('ripgrep-14.1.1-x86_64-unknown-linux-musl.tar.gz',
         '4cf9f2741e6c465ffdb7c26f38056a59e2a2544b51f7cc128ef28337eeae4d8e', 'rg'))
    # Hashes published alongside the upstream 14.1.1 release assets.
    url = 'https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/'+asset
    raw = fetch(url)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('ripgrep_archive_digest_mismatch')
    folder = asset.removesuffix('.zip').removesuffix('.tar.gz')
    if os.name == 'nt':
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            binary = archive.read(folder+'/'+executable)
    else:
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            binary = archive.extractfile(folder+'/'+executable).read()
    destination = Path(os.environ['RUNNER_TEMP'])/'tare-test-ripgrep-14.1.1'
    destination.mkdir(exist_ok=True)
    target = destination/executable
    target.write_bytes(binary)
    target.chmod(0o755)
    with open(os.environ['GITHUB_PATH'], 'a', encoding='utf-8') as stream:
        stream.write(str(destination.resolve())+'\n')
    print('Qualified test dependency:', asset, digest)


if __name__ == '__main__':
    main()
