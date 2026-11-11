"""Text-only facade around the captured native95 backend and core session.

Kept outside ripple_native95: its recursive native Source identity must not change.
"""
from copy import deepcopy


def _request(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("native_request must be a dictionary")
    result = deepcopy(value)
    if result.pop("images", None) is not None:
        raise ValueError("Ripple native95 text does not accept image references")
    return result


def _no_images(images):
    if images is not None:
        raise ValueError("Ripple native95 text does not accept image references")


def _open_original(artifact, *, options, strength, native_request):
    from particle_sliders.inference import open_inference
    from ripple_native95.factory import INFERENCE
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

    def render(self, *, strength=1., seed=0, images=None, output_type="text", **native):
        _no_images(images)
        if output_type != "text":
            raise ValueError("Ripple native95 produces text")
        request = _request(native)
        session = self._session
        session._open()
        session.capabilities.check("render", strength=strength, references=False,
                                   output_type="text")
        session._preflight("render", strength, False, "text", request)
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
        raise ValueError("Ripple native95 produces text")
    request = _request(native_request)
    session = _open_original(artifact, options=options, strength=strength,
                             native_request=request)
    return TextInferenceSession(session)
