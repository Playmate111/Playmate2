import importlib


def parallelize_transformer(transformer, *args, **kwargs):
    transformer_cls_name = transformer.__class__.__name__
    if False:
        pass
    elif transformer_cls_name.startswith("Wan"):
        adapter_name = "wan"
    else:
        raise ValueError(f"Unknown transformer class name: {transformer_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_transformer_fn = getattr(adapter_module, "parallelize_transformer")
    return parallelize_transformer_fn(transformer, *args, **kwargs)


def parallelize_pipe(pipe, *args, **kwargs):
    adapter_name = "wan"

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    parallelize_pipe_fn = getattr(adapter_module, "parallelize_pipe")
    return parallelize_pipe_fn(pipe, *args, **kwargs)
