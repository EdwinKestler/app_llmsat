from .config import PipelineConfig, load_config

# Lazy import to avoid circular dependency (pipeline.py → downloader → pipeline.config)


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run
    return _run(*args, **kwargs)


def run_detection(*args, **kwargs):
    from .geodetector import run_detection as _run
    return _run(*args, **kwargs)
