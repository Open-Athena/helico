import json
from pathlib import Path

import pytest

from helico.checkpoint_storage import publish_checkpoint, publish_json


INCOMPLETE = OSError(22, "You did not provide the number of bytes specified by the Content-Length HTTP header. Received: 0, Expected: 52428800")


class Storage:
    def __init__(self, upload_errors=(), pointer_errors=(), truncate=False):
        self.upload_errors = iter(upload_errors)
        self.pointer_errors = iter(pointer_errors)
        self.objects = {"run/latest.json": b'{"step": 9000}'}
        self.uploads, self.pointers, self.truncate = [], 0, truncate

    def put_file(self, path, uri):
        content = Path(path).read_bytes()
        self.uploads.append(content)
        error = next(self.upload_errors, None)
        if error:
            raise error
        self.objects[uri] = content[:-1] if self.truncate else content

    def size(self, uri):
        return len(self.objects[uri])

    def pipe_file(self, uri, content):
        self.pointers += 1
        error = next(self.pointer_errors, None)
        if error:
            raise error
        self.objects[uri] = content


def publish(fs, tmp_path):
    path = tmp_path / "step-009250.pt"
    path.write_bytes(b"complete checkpoint bytes")
    return publish_checkpoint(fs, path, "run", 9250, "data-lock", sleep=lambda _: None)


def test_incomplete_upload_retries_full_bytes_then_advances_pointer(tmp_path):
    fs = Storage(upload_errors=[INCOMPLETE])
    uri = publish(fs, tmp_path)
    assert fs.uploads == [b"complete checkpoint bytes"] * 2
    assert fs.objects[uri] == b"complete checkpoint bytes"
    assert json.loads(fs.objects["run/latest.json"]) == {
        "step": 9250, "checkpoint": uri, "data_lock_sha256": "data-lock",
        "sha256": __import__("hashlib").sha256(b"complete checkpoint bytes").hexdigest(),
        "bytes": len(b"complete checkpoint bytes")}


def test_pointer_retry_does_not_repeat_checkpoint_upload(tmp_path):
    fs = Storage(pointer_errors=[INCOMPLETE])
    publish(fs, tmp_path)
    assert len(fs.uploads) == 1 and fs.pointers == 2


def test_evaluation_metadata_recovers_incomplete_body():
    fs = Storage(pointer_errors=[INCOMPLETE])
    record = dict(status='complete', artifact='result.pt.gz', artifact_bytes=20)
    publish_json(fs, 'eval/target.json', record, sleep=lambda _: None)
    assert fs.pointers == 2
    assert json.loads(fs.objects['eval/target.json']) == record


@pytest.mark.parametrize('errors,exception,count', [
    ([INCOMPLETE] * 4, OSError, 3),
    ([PermissionError('AccessDenied')], PermissionError, 1),
])
def test_failed_evaluation_metadata_does_not_mark_complete(errors, exception, count):
    fs = Storage(pointer_errors=errors)
    with pytest.raises(exception):
        publish_json(fs, 'eval/target.json', dict(status='complete'), sleep=lambda _: None)
    assert fs.pointers == count
    assert 'eval/target.json' not in fs.objects


@pytest.mark.parametrize("storage,exception,uploads", [
    (lambda: Storage(upload_errors=[INCOMPLETE] * 4), OSError, 3),
    (lambda: Storage(upload_errors=[PermissionError("AccessDenied")]), PermissionError, 1),
    (lambda: Storage(truncate=True), RuntimeError, 1),
])
def test_failed_upload_never_advances_resume_pointer(tmp_path, storage, exception, uploads):
    fs = storage()
    with pytest.raises(exception):
        publish(fs, tmp_path)
    assert len(fs.uploads) == uploads
    assert fs.pointers == 0
    assert json.loads(fs.objects["run/latest.json"]) == {"step": 9000}
