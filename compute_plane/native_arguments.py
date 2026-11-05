"""Normalize text-only tool formats without broadening owner schemas."""


def normalize_booleans(args, schema):
    """Copy arguments, converting only exact text booleans on boolean properties.

    Unknown fields and invalid values are left for the owner's validator; text
    content and caller-owned dictionaries are never changed.
    """
    if not isinstance(args, dict):
        return args
    result = dict(args)
    for key, spec in schema.get('properties', {}).items():
        if spec.get('type') == 'boolean' and isinstance(result.get(key), str):
            if result[key] == 'true':
                result[key] = True
            elif result[key] == 'false':
                result[key] = False
    return result
