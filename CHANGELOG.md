# Changelog

Notable changes to `tare.tools.kernel` are recorded here, newest first.

## Unreleased

- CI runs on GitHub-hosted Linux and Windows runners (`Tests (linux)`, `Tests (windows)`) with the README's own install commands: the runtime and SpecGraph come from their public repositories at the pinned commits, without deploy keys or a hand-made environment. The repository variables `CI_RUNNER_LINUX` and `CI_RUNNER_WINDOWS` move it to your own runners, and pull requests from forks then skip it. The README installs the runtime and SpecGraph at the commits CI pins; the ones it named are not published.
- `test_default_read_returns_on_first_output` keeps its child running for 60 s instead of 1 s: on a loaded runner the read could start after the child had exited.
- CI pins the agent runtime and SpecGraph at the first commits of their public repositories; the commits it pinned before are not published.
- First public release. The repository starts from a single commit; earlier history is not published. It contains the
  kernel described in the README (provider adapters, mediated tools, Git workspaces, terminal sessions, hooks and
  append-only session journals), its specifications and contracts, among them the GPU lease contract, and the test
  suite.
