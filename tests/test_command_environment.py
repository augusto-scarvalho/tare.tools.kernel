import json
import os
import subprocess

import pytest
from compute_plane.command_environment import describe
from compute_plane.terminal_host import TerminalHost, TerminalClient


def test_context_uses_granted_path_without_dumping_secrets(tmp_path):
    with TerminalHost() as host:
        env=dict(os.environ, TOP_SECRET='never-in-model-context')
        client=TerminalClient(host.bind(owner='test',repositories={'p':str(tmp_path)},env=env))
        context=client.request('describe')['execution_environment']
    assert 'never-in-model-context' not in json.dumps(context)
    assert 'PATH' not in context and 'TOP_SECRET' not in context
    assert len(json.dumps(context)) < 4096
    assert context['tools']
    assert 'never restart' in context['command_rules']


@pytest.mark.skipif(os.name!='nt',reason='Windows PowerShell command contract')
def test_advertised_powershell_runs_literal_spaced_unicode_path_and_propagates_exit(tmp_path):
    context=describe(dict(os.environ))
    assert context['platform']=='Windows' and context['preferred_shell'] in ('pwsh','powershell')
    project=tmp_path/'project with spaces [1]'; project.mkdir()
    source=project/'ação [um].txt'; source.write_text('café São Paulo',encoding='utf-8')
    script=(context['shell_script_prefix']+'$ErrorActionPreference="Stop"; '
        "Get-Content -LiteralPath '"+str(source).replace("'","''")+"' -Encoding UTF8; exit 7")
    result=subprocess.run([*context['shell_argv'],script],cwd=project,capture_output=True,timeout=10)
    assert result.returncode==7 and result.stdout.decode('utf-8').strip()=='café São Paulo'


def test_empty_path_does_not_advertise_unavailable_shells():
    context=describe({'PATH':''})
    assert context['preferred_shell'] is None and context['shell_argv']==[]
