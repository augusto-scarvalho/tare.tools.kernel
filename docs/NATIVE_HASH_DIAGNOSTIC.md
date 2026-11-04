# Invalid edit hashes and nullable arguments

Kernel 0.9.5 rejects malformed hashes with `invalid_sha256` before observing
source bytes or starting an effect. The message requests the lowercase SHA-256
returned by `read`, or JSON null for missing-file creation. It never converts
`"null"` to null. A valid stale hash still returns `source_changed_read_again`;
null cannot overwrite an existing file. The native edit description states the
same distinction. Delete and rename share the mediated hash validation.

The lifecycle Qwen failure exposed a misleading diagnostic, but the route also
has a separate representational problem. The live Qwen template with SHA-256
`12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`
renders identical prompts for a tool argument whose JSON value is null and one
whose value is the string `"null"`. Two `/apply-template` requests confirmed the
collision without inference. Their identical prompt SHA-256 was
`837ba39b7090cdf34616caffc19349dccab86822f37f13c03a6c1d5c6d287662`.

The source of deployed slop build `b10165-71676e46c` selects a string-value parser
whenever `common_schema_info::resolves_to_string` finds a string alternative,
including `["string", "null"]`. The mapper then quotes the parameter as a string.
The current local slop source retains this behavior. These facts establish a
route limitation; historical raw model output was not captured, so they do not
reconstruct the precise original emission. Kernel Chat and Converse regression
fixtures confirm they preserve JSON null and string values without coercion.

This release fixes the diagnostic, not the model-server parser or template.
Neither a provider-specific coercion nor a changed nullable schema silently
repairs that protocol. The successor must define a representation preserving
both values, test template replay and complete/streamed parsing, and qualify
creation, exact verification and continuation before repeating the long Qwen
lifecycle case. Existing journals retain their original failed outcomes.
