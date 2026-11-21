"""Text-only facade around the captured native backend and core session.

Kept outside ripple_native: its recursive native Source identity must not change.
"""
from copy import deepcopy


def _request(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("native_request must be a dictionary")
    if "steering" in value:
        raise ValueError("Supply activation steering to render, after opening the original RAW owner")
    result = deepcopy(value)
    if result.pop("images", None) is not None:
        raise ValueError("Ripple native text does not accept image references")
    return result


def _no_images(images):
    if images is not None:
        raise ValueError("Ripple native text does not accept image references")


def _open_original(artifact, *, options, strength, native_request):
    from particle_sliders.inference import open_inference
    from ripple_native.factory import INFERENCE
    return open_inference(artifact, options=options, backend=INFERENCE,
                          operation="render", strength=strength, output_type="text",
                          native_request=native_request)


class TextInferenceSession:
    """Own the original session; observation and retryable close stay in core."""
    def __init__(self, session):
        self._session = session

    @property
    def selection(self):
        return self._session.selection

    @property
    def manifest(self):
        return self._session.manifest

    @property
    def closed(self):
        return self._session.closed

    def render(self, *, strength=1., seed=0, images=None, output_type="text", steering=None, **native):
        _no_images(images)
        if output_type != "text":
            raise ValueError("Ripple native produces text")
        request = _request(native)
        gains = request.pop("step_gains", None)
        if steering is not None and gains is not None:
            raise ValueError("Choose either activation steering or fixed step gains")
        session = self._session
        session._open()
        session.capabilities.check("render", strength=strength, references=False,
                                   output_type="text")
        session._preflight("render", strength, False, "text", request)
        if gains is not None or steering is not None:
            native_owner = getattr(session.binding.response, "__self__", None)
            if getattr(native_owner, "network", None) is not session.binding.network:
                raise ValueError("Scheduled strength requires the captured Ripple native owner")
            with session._observation(seed, request):
                if steering is not None:
                    from .adaptive_text import generate
                    return generate(native_owner, strength=strength, seed=seed, output_type="text",
                                    steering=steering, **request)
                from .scheduled_text import generate
                return generate(native_owner, strength=strength, seed=seed, output_type="text",
                                step_gains=gains, **request)
        # Original successful text caller: omit images rather than forwarding None.
        with session._observation(seed, request):
            return session.binding.render(strength=strength, seed=seed,
                                          output_type="text", **request)

    def close(self):
        return self._session.close()

    def __enter__(self):
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        return self._session.__exit__(exc_type, exc, traceback)


def open_text_inference(artifact, *, options=None, strength=1., images=None,
                        output_type="text", native_request=None):
    """Open the unchanged native-only RAW artifact; no game/teachers/initialization."""
    _no_images(images)
    if output_type != "text":
        raise ValueError("Ripple native produces text")
    request = _request(native_request)
    gains = request.pop("step_gains", None)
    if gains is not None:
        from ripple_native.finished_text import validate_request
        from .scheduled_text import step_strengths
        validate_request(strength=strength, output_type="text", **request)
        step_strengths(gains, strength=strength, max_new_tokens=request.get("max_new_tokens", 64))
    session = _open_original(artifact, options=options, strength=strength,
                             native_request=request)
    return TextInferenceSession(session)
