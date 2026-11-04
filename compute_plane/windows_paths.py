r"""Windows' realpath can keep the `\\?\` long-path prefix; tare strips it where the plain path names the same file.

CPython's ntpath.realpath looks a path up twice and keeps the `\\?\` prefix when the second lookup fails
differently from the first: a scanner or another writer holding a just-created folder is enough. The prefixed
path then leaks into comparisons, receipts and Git arguments ("could not create leading directories of
'//?/C:/...': Invalid argument"). Installed once when compute_plane is imported, so every tare process gets it.
ponytail: a process-wide wrapper of the stdlib function; drop it once CPython stops keeping the prefix there.
"""
import ntpath
import os
from pathlib import PureWindowsPath

PREFIX = '\\\\?\\'
RESERVED = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$', *(f'{d}{i}' for d in ('COM', 'LPT') for i in range(1, 10))}


def plain(result, requested):
    """`result` without the prefix when the plain path means the same file: a drive path the caller did not
    prefix itself, shorter than MAX_PATH, without reserved device names or names ending in a dot or space
    (which the plain form would rewrite)."""
    if (not isinstance(result, str) or not result.startswith(PREFIX) or result[5:6] != ':'
            or (isinstance(requested, str) and requested.startswith(PREFIX))):
        return result
    path = result[len(PREFIX):]
    names = PureWindowsPath(path).parts[1:]
    if (len(path) >= 260 or any(name.split('.')[0].rstrip(' ').upper() in RESERVED for name in names)
            or any(name.endswith(('.', ' ')) for name in names)):
        return result
    return path


def install():
    if os.name != 'nt' or getattr(ntpath.realpath, 'tare_plain', False):
        return
    original = ntpath.realpath

    def realpath(path, *args, **kwargs):
        return plain(original(path, *args, **kwargs), os.fspath(path))

    realpath.tare_plain, realpath.__wrapped__ = True, original
    ntpath.realpath = realpath  # os.path is ntpath here; pathlib looks it up at each resolve()
