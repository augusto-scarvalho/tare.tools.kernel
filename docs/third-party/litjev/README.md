# LitJEV attribution

`compute_plane/litjev_choice.py` adapts the prompt wording and single-question
candidate scoring method from zhengxuyu/litjev, revision
`e7fb109a7466da9709028eb9c4e9f16eaeb4e2a3` (Apache-2.0).
Original license and NOTICE accompany this file. Tare's changes implement a
native HTTP readout, token-boundary checks, bounded responses and truthful
one-token accounting; they do not reproduce the upstream Transformers backend.
