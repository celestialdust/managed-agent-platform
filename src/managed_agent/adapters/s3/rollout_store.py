"""The Rollout's object-store adapter: one bucket, one object per Session.

Overwriting in place is safe here in a way it would not be for Evidence: an S3 PUT
either replaces the object or leaves the previous one whole, so no reader ever sees
half a Rollout. That is what lets the key carry no Turn number -- the only version
anyone will ever ask for is the newest one.

**What this class does not claim is that no history accumulates.** The provisioned
bucket has versioning enabled, so every ship-out leaves a noncurrent version behind:
one thread-length-sized copy per completed Turn per Session, kept until something
expires it. Nothing here deletes, and a noncurrent-version lifecycle rule is not in
this repository's Terraform. That is a recorded cost, not a mitigated one.

An absent key comes back as None rather than as an exception. A Session that has
completed no Turn is the ordinary first state of every Session, not a fault, and making
the caller catch something for it would put a normal path behind an except block.

Only `NoSuchKey` reads as absence, and the narrowness is deliberate. `get_object` on a
bucket whose policy withholds `s3:ListBucket` answers `AccessDenied` for a key that is
merely absent, and mapping that to None would report "this Session has completed no
Turn" for a misconfigured deployment -- the one answer that makes recovery silently
restore nothing. Anything that is not `NoSuchKey` propagates.
"""

import aioboto3  # type: ignore[import-untyped]


class S3RolloutStore:
    """Reads and writes one bucket. The key layout belongs to the caller, not here."""

    def __init__(self, session: aioboto3.Session, bucket: str) -> None:
        self._session = session
        self._bucket = bucket

    async def put(self, key: str, body: bytes) -> None:
        async with self._session.client("s3") as s3:
            await s3.put_object(Bucket=self._bucket, Key=key, Body=body)

    async def get(self, key: str) -> bytes | None:
        """The stored bytes, or None when this key was never written.

        The body is read **inside** the `async with`: botocore's streaming body is bound
        to the client that produced it, so reading after the client closes raises.
        """
        async with self._session.client("s3") as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except s3.exceptions.NoSuchKey:
                return None
            body: bytes = await response["Body"].read()
        return body
