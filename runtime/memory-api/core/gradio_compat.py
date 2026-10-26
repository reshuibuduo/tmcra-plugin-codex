from __future__ import annotations


def patch_gradio_schema_bool_support() -> None:
    """
    Work around a Gradio 4.44.1 bug where nested JSON schema values like
    `additionalProperties: true` are passed into `_json_schema_to_python_type()`
    as booleans, but the upstream helper assumes every schema node is a dict.
    """
    try:
        from gradio_client import utils as client_utils
    except Exception:
        return

    if getattr(client_utils, "_tmcra_bool_schema_patch", False):
        return

    original = client_utils._json_schema_to_python_type

    def patched(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original(schema, defs)

    client_utils._json_schema_to_python_type = patched
    client_utils._tmcra_bool_schema_patch = True
