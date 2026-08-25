"""The Rollout adapter over a fake S3 client held in this process.

The fake stands in at the **constructor**, so what runs is the shipped class:
`put_object`, `get_object` and the `NoSuchKey` arm are all executed here, not merely
type-checked. That is more than the sibling adapter in this package gets --
`tests/control/test_file_upload_download.py`'s `shipped_storage` docstring records
`S3UploadedFiles`'s object half as "structurally checked and never executed" -- and it
is possible only because `aioboto3` is untyped, so `aioboto3.Session` resolves to `Any`
and a fake satisfies the annotation under `mypy --strict`.

What it cannot say: no test here reaches a real bucket. That the IAM policy permits
`PutObject` under `rollouts/`, and that a real `get_object` on an absent key answers
`NoSuchKey` rather than `AccessDenied`, are both untested. The MinIO fixture that would
settle them is deferred and uses `testcontainers.core.container.DockerContainer`
directly, so it needs no new dependency when it comes back.
"""

from __future__ import annotations

from types import TracebackType
from typing import Self

import pytest

from managed_agent.adapters.s3.rollout_store import S3RolloutStore


class BucketRefused(Exception):
    """Stands in for an AccessDenied: a refusal that is not an absent key."""


class _NoSuchKey(Exception):
    """What botocore generates from the service model, as far as this test cares."""


class _Exceptions:
    NoSuchKey = _NoSuchKey


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self) -> bytes:
        return self._body


class _Client:
    exceptions = _Exceptions

    def __init__(self, objects: dict[str, bytes], refused: frozenset[str]) -> None:
        self._objects = objects
        self._refused = refused

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self._objects[Key] = Body

    async def get_object(self, *, Bucket: str, Key: str) -> dict[str, _Body]:
        if Key in self._refused:
            raise BucketRefused(Key)
        if Key not in self._objects:
            raise _NoSuchKey(Key)
        return {"Body": _Body(self._objects[Key])}


class FakeS3Session:
    """`.client("s3")` and nothing else, because that is all the adapter calls."""

    def __init__(self, refused: frozenset[str] = frozenset()) -> None:
        self.objects: dict[str, bytes] = {}
        self._refused = refused

    def client(self, name: str) -> _Client:
        assert name == "s3"
        return _Client(self.objects, self._refused)


async def test_a_put_then_a_get_returns_the_same_bytes() -> None:
    fake = FakeS3Session()
    store = S3RolloutStore(fake, "a-bucket")

    await store.put("rollouts/one", b"first")
    assert await store.get("rollouts/one") == b"first"


async def test_a_second_put_replaces_the_object_rather_than_adding_one() -> None:
    """The key carries no Turn, so replacement is the whole storage model."""
    fake = FakeS3Session()
    store = S3RolloutStore(fake, "a-bucket")

    await store.put("rollouts/one", b"first")
    await store.put("rollouts/one", b"second")

    assert await store.get("rollouts/one") == b"second"
    assert list(fake.objects) == ["rollouts/one"]


async def test_a_key_never_written_reads_as_absent_rather_than_raising() -> None:
    assert await S3RolloutStore(FakeS3Session(), "a-bucket").get("nothing") is None


async def test_two_keys_do_not_collide() -> None:
    fake = FakeS3Session()
    store = S3RolloutStore(fake, "a-bucket")

    await store.put("rollouts/one", b"one")
    await store.put("rollouts/two", b"two")

    assert await store.get("rollouts/one") == b"one"
    assert await store.get("rollouts/two") == b"two"


async def test_a_refusal_that_is_not_an_absent_key_propagates() -> None:
    """Mapping AccessDenied to None would report a misconfigured bucket as a Session
    that has completed no Turn, which is the one answer that restores nothing."""
    store = S3RolloutStore(FakeS3Session(refused=frozenset({"walled"})), "a-bucket")
    with pytest.raises(BucketRefused):
        await store.get("walled")
