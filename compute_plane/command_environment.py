"""Small host facts for commands; no environment dump or executable probes."""
import os
import shutil
import sys


def describe(env):
    windows = os.name == 'nt'
    names = ('pwsh', 'powershell', 'cmd', 'python', 'git', 'rg', 'node', 'npm') if windows else (
        'bash', 'sh', 'python3', 'python', 'git', 'rg', 'node', 'npm')
    tools = {name: path for name in names if (path := shutil.which(name, path=env.get('PATH', '')))}
    shell = next((name for name in (('pwsh', 'powershell', 'cmd') if windows else ('bash', 'sh')) if name in tools), None)
    if shell in ('pwsh', 'powershell'):
        argv = [tools[shell], '-NoLogo', '-NoProfile', '-NonInteractive', '-Command']
        # Windows PowerShell may start its provider in System32 when cwd has
        # wildcard characters. The process cwd still retains the literal path.
        prefix = ('Set-Location -LiteralPath ([Environment]::CurrentDirectory) -ErrorAction Stop; '
                  '[Console]::OutputEncoding = $OutputEncoding = [System.Text.UTF8Encoding]::new($false); ')
    elif shell == 'cmd':
        argv, prefix = [tools[shell], '/d', '/c'], 'chcp 65001 >nul & '
    elif shell:
        argv, prefix = [tools[shell], '-c'], ''
    else:
        argv, prefix = [], ''
    return {'platform': 'Windows' if windows else sys.platform, 'tools': tools,
        'host_python': sys.executable, 'preferred_shell': shell, 'shell_argv': argv,
        'shell_script_prefix': prefix, 'output_encoding': 'utf-8',
        'command_rules': (
            'terminal_start executes argv directly, without an implicit shell. '
            'Pass each argument as one JSON string; do not add shell quotes around argv elements. '
            'The selected repository determines cwd. Use absolute tool paths above when available. '
            'For shell syntax append one script string to shell_argv; prepend shell_script_prefix for UTF-8 output. '
            + ('Use Windows paths and PowerShell syntax with the preferred PowerShell. Do not assume Bash/WSL. '
               'Windows PowerShell 5.1 does not support && or ||. For file cmdlets use absolute -LiteralPath '
               '(relative paths can silently yield no results when cwd contains brackets). '
               'Read UTF-8 project files with -Encoding UTF8; console encoding does not set file encoding. '
               "Example: Get-Content -LiteralPath (Join-Path (Get-Location).ProviderPath 'file.txt') -Encoding UTF8. "
               'After Set-Location, Environment.CurrentDirectory can still name the original process directory. '
               'For PowerShell failures use $ErrorActionPreference="Stop"; after a native command propagate '
               'its exit code with exit $LASTEXITCODE. Invoke .cmd/.bat tools through cmd /d /c. '
               if windows else '') +
            'For finite commands read the same handle with wait_for_exit=true and wait_ms=30000 until it ends. '
            'A read wait expiring while status=running is not a command timeout; continue reading, never restart it. '
            'Process timeout defaults to 300 seconds; choose an adequate timeout_seconds or null for a persistent service. '
            'Inspect exit_code and both output streams. A start acknowledgement is not command success. '
            'After a process timeout, effects may already exist: inspect them before any retry.')}
