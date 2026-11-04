<div align="center">

# tare.tools.kernel

**Execução limitada de modelos e ferramentas para o ecossistema tare.tools: adaptadores de provedores, ferramentas mediadas, workspaces Git, sessões de terminal, hooks e journals de sessão append-only.**

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://python.org)

<p align="center">
  <a href="#o-que-é-o-taretoolskernel">O que é o tare.tools.kernel</a> •
  <a href="#o-que-existe-hoje">O que existe hoje</a> •
  <a href="#instalar-e-testar">Instalar e testar</a> •
  <a href="#compute-plane--limites-de-execução">Limites de execução</a> •
  <a href="#início-rápido-da-api-python">Início rápido da API</a> •
  <a href="#família-do-ecossistema">Família do ecossistema</a> •
  <a href="CHANGELOG.md">Changelog</a> •
  <a href="#licença">Licença</a>
</p>

<p align="center">
  <em>🇺🇸 English version: <a href="README.md">README.md</a>.</em>
</p>

</div>

---

## O que é o tare.tools.kernel?

O `tare.tools.kernel` executa chamadas de modelo e seus efeitos para o orquestrador
`tare.tools` ([tare.tools.os](https://github.com/augusto-scarvalho/tare.tools.os)),
que é dono da seleção de Work, da revisão, da incorporação e das interfaces do
operador. O OS fixa este repositório como submódulo Git em
`satellites/tare.tools.kernel` e o instala a partir dali; sozinho, o Kernel é uma
biblioteca Python mais o comando `tare-hook`.

Os [adaptadores nativos de modelo](docs/guides/NATIVE_MODEL_ADAPTERS.md) fornecem
portas de provedor e de processo ao
[`tare.tools.agent-runtime`](https://github.com/augusto-scarvalho/tare.tools.agent-runtime),
o motor tático atual. Os guias removidos do Aider e do driver agregado de sessão
HTTP não são mais distribuídos ([conteúdo do repositório](docs/guides/REPOSITORY_CONTENT.md)).

## O que existe hoje

O código implementado está em `compute_plane/`, mais os recibos de produtor em `assurance_plane/`:

| Área | Módulos |
|---|---|
| Adaptadores de modelo | `native_models.py` (codecs HTTP, projeção de contexto), `native_cli.py` (CLIs Claude/Codex/AGY), `native_stream.py`, `native_usage.py` |
| Ferramentas mediadas e efeitos | `mediated_tools.py`, `native_capabilities.py`, `native_effect_recovery.py`, `tool_hooks.py`, `hook_observations.py` |
| Journal de sessão | `native_journal.py`: JSONL append-only por sessão, usado em replay e recuperação |
| Workspaces Git | `workspace.py` (worktrees vinculadas com leases exatos), `candidate_git.py` |
| Processos e terminais | `managed_process.py`, `terminal_sessions.py`, `terminal_pty.py`, `terminal_host.py` |
| Confinamento opcional no Windows | `windows_mxc.py`, `codex_sandbox.py`, `confined_workspace.py` ([guia](docs/guides/CONFINED_EXECUTION.md)) |
| Recibos de produtor | `assurance_plane/producer_receipts.py`: recibos Ed25519 que vinculam um produtor aos bytes de saída |
| Contexto | `native_instructions.py`, `project_context_hooks.py` |

Limites, sem rodeios:

- `control_plane/`, `data_plane/` e `experience_plane/` são pacotes vazios do
  plano original de cinco planos ([SPEC-KERNEL-001](docs/SPEC-KERNEL-001.md)).
  Não há store SQLite WAL, notário de incorporação nem cockpit aqui; incorporação,
  leases, cockpit e TUI ficam no tare.tools.os.
- Raízes de projeto selecionam diretórios de trabalho; **não são sandbox de comandos**.
  Worktrees vinculadas isolam mudanças Git; os perfis opcionais de confinamento
  acima existem só no Windows e ficam desligados a menos que um host os injete.

## Instalar e testar

Você precisa de Python 3.11 ou mais recente, Git e
[ripgrep](https://github.com/BurntSushi/ripgrep) (`rg` no `PATH`; os testes de
busca o executam). O Kernel depende do `tare.tools.agent-runtime`, e seus testes
de hooks usam o `tare.tools.specgraph`; nenhum dos dois está no PyPI, então
instale-os do GitHub primeiro, nas revisões contra as quais a CI testa
(`RUNTIME_REVISION` e `SPECGRAPH_REVISION` em
[`.github/workflows/ci.yml`](.github/workflows/ci.yml); use os valores de lá se
tiverem mudado):

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.kernel.git
cd tare.tools.kernel
python -m venv .venv
. .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install "git+https://github.com/augusto-scarvalho/tare.tools.agent-runtime@133d9bba5c584c1959595a50e86430670bf666f2"
python -m pip install "git+https://github.com/augusto-scarvalho/tare.tools.specgraph@cce9edde05950bfcce980a1634f974d48308437c"
python -m pip install -e ".[dev,attachments,live-cache]"
python -m pytest tests/ -q
```

Os testes rodam offline: não chamam nenhum provedor de modelo e não baixam nada.
A CI roda esses mesmos comandos em runners Linux e Windows hospedados pelo GitHub. Para
rodá-los nas suas próprias máquinas, defina as variáveis de repositório `CI_RUNNER_LINUX`
e `CI_RUNNER_WINDOWS` com uma lista de labels de runner (por exemplo `["self-hosted","Linux"]`).
As regras para quem contribui estão em [AGENTS.md](AGENTS.md).

---

## Compute Plane & Limites de Execução

### Hooks canônicos de fornecedor

O comando leaf `tare-hook` aplica uma única política de lifecycle a Claude
Code, Codex e Google Antigravity:

```text
tare-hook claude
tare-hook codex
tare-hook agy
```

Cada adaptador de fornecedor apenas traduz seu envelope JSON nativo. O gate
compartilhado intercepta `Grep`/`grep_search` nativos e `rg`/`grep` no shell e
direciona toda busca no repositório para
`specgraph resolve <consulta> --format json`. Esse resolver tenta a estrutura
primeiro, libera recuperação literal limitada somente em `MISS`, `STALE`,
`TIMEOUT` ou `UNAVAILABLE` e depois indexa a evidência validada para reuso.
Enquanto o servidor local de modelos responde em `/health`, o mesmo gate nega a
criação de subagentes de fronteira (Claude `Agent`/`Task`, Codex `spawn_agent`) e
indica ao chamador `tare-ops explore`/`evidence`/`research`; um servidor
inacessível falha aberto, e `TARE_LOCAL_MODELS_BYPASS=1` é a exceção explícita do
usuário. Configure `TARE_LOCAL_MODELS` no ambiente que inicia a CLI do fornecedor
com a URL do gateway do operador; o padrão portátil é `http://127.0.0.1:8080`.
Entradas inválidas ou grandes demais no pré-hook são negadas antes da ferramenta.
Edições reconhecidas também recebem contexto consultivo e limitado de impacto
antes da edição e uma avaliação de desatualização depois do sucesso; esses avisos
nunca reconstroem o cache do SpecGraph nem concedem autoridade.

Exemplos de configuração local ficam em
[`docs/CANONICAL_TOOL_HOOKS.md`](docs/CANONICAL_TOOL_HOOKS.md).
O próprio Kernel também prepara os três manifests e um `specgraph.yaml`
explícito como primeiro dogfood local; nenhuma configuração global dos
fornecedores é alterada. O `search_paths` neutro torna README, changelog e
documentação comum pesquisáveis sem classificá-los como especificações ou ADRs
formais. A expansão continua opt-in e deve reutilizar esse limite qualificado.

Execuções gerenciadas também podem usar
`compute_plane.backlog_work.materialize_backlog_work` para obter o envelope
exato e limitado de `backlog ground`, seguido de `check_backlog_scope` nos
checkpoints do runtime. O Kernel apenas transporta e observa essa cerca somente
leitura; o host converte `CURRENT`, `DRIFTED` ou `UNAVAILABLE` no work guard
injetado do Agent Runtime. Ele não escreve no backlog nem concede Authority.

Veja [hooks de contexto de projeto com escopo](docs/SCOPED_PROJECT_CONTEXT_HOOKS.md)
para conselhos opt-in do SpecGraph, observações duráveis de edição, diagnóstico
limitado por host e os bytes exatos de recibo expostos pelo progresso de Work
gerenciado.

`compute_plane.managed_work` fornece o serializer opt-in do caminho estrito e o
substrato de observação de progresso. Ele coloca Work imutável antes da
instrução do operador, roda apenas os comandos de verificação fornecidos
explicitamente no workspace efêmero, calcula hashes de arquivos rastreados
limitados e devolve contagens e identidades de digest. Ele não interpreta texto
do modelo nem liquida Evidence.

A execução gerenciada qualifica uma worktree Git vinculada e limpa e
inicia os processos filhos com um ambiente de credenciais mínimas. Esse é um
limite explícito, não uma alegação de isolamento rígido de filesystem ou rede
em todo host suportado.

O laboratório P2 de cache de prompts também fica no Compute Plane. Seu adaptador
de provedor falso consome uma projeção admitida e destacada do Agent Runtime e
produz os bytes exatos da requisição e um `PromptCachePlan` imutável. O cache é
limitado, em memória, descartável e não usa rede nem SDK de provedor; sua
telemetria é explicitamente não autoritativa. O cache com provedor real continua
sendo uma fatia vertical P3 separada.

A fatia P3a autorizada pelo operador acrescenta adaptadores live deliberadamente
estreitos: OpenRouter Chat Completions com provedor fixado, Amazon Bedrock
`InvokeModel` com body Anthropic exato e Bedrock Mantle Responses com breakpoint
explícito do GPT-5.6 e `store=false`. Eles preservam a correção com cache frio,
reportam leituras e escritas de cache, latência e estimativas de custo e
continuam caminhos opt-in de qualificação, não padrões de produção.

A inferência passa pelos adaptadores nativos (`native_models.py` para provedores HTTP e
`native_cli.py` para CLIs oficiais). Os antigos drivers Aider e LittleCoder foram
aposentados; seleção e validação do modelo ficam fora dos adaptadores.

---

## Início rápido da API Python

A porta nativa de capacidades aceita uma lista explícita `capabilities`,
incluindo a busca opt-in `search`. A busca usa o ripgrep instalado no host, com
operandos de arquivo validados e com escopo e consultas literais ou regex; ela
não exige uma consulta prévia ao SpecGraph nem concede um shell. Ela informa
caminhos, números de linha, cobertura e truncamento. Os papéis continuam
limitando a seleção, e uma seleção omitida preserva os padrões anteriores. As
verificações de caminho em leitura e edição são fronteiras cooperativas do
filesystem; executar um verificador nomeado não é sandbox de filesystem para o
código candidato.

```python
from pathlib import Path

from compute_plane.workspace import qualify_ephemeral_git_worktree

qualification = qualify_ephemeral_git_worktree(Path("/tmp/worktree_01"))
print(qualification.head)
```

---

## Família do ecossistema

| Repositório | Papel | Tecnologia |
|---|---|---|
| **[`tare.tools.os`](https://github.com/augusto-scarvalho/tare.tools.os)** | Agent Operating System & Central Swarm Orchestrator; serviço do nó de GPU (`node/`): gateway de modelos, lease compartilhado da GPU | Python, AsyncIO, Submodules |
| **[`tare.tools.kernel`](https://github.com/augusto-scarvalho/tare.tools.kernel)** | Execução limitada de modelos e ferramentas (este repo) | Python, Git Worktrees, ConPTY/PTY |
| **[`tare.tools.agent-runtime`](https://github.com/augusto-scarvalho/tare.tools.agent-runtime)** | Loop tático de agente delimitado | Python (só stdlib) |
| **[`tare.tools.specgraph`](https://github.com/augusto-scarvalho/tare.tools.specgraph)** | Spec-Driven Development & Causal Matrix | Python, Tree-Sitter, CPG AST |
| **[`tare.tools.backlog-graph`](https://github.com/augusto-scarvalho/tare.tools.backlog-graph)** | Deterministic DAG Task Backlog Engine | Python (Pure Stdlib), CAS |
| **[`tare.tools.library`](https://github.com/augusto-scarvalho/tare.tools.library)** | Biblioteca técnica e registros de conhecimento | Markdown, JSON |
| **[`tare.tools.dialog-engine`](https://github.com/augusto-scarvalho/tare.tools.dialog-engine)** | Topological Interaction & AST Dialog Graphs | Python, Statecharts |

---

## Licença

Licenciado sob a Apache License, Version 2.0. Veja [LICENSE](LICENSE) para os detalhes.
