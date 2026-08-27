"""A real TLS handshake between a real server and the control plane's real dial context.

Every other case about certificates in this tree reads fields off a certificate. None of
them proves the two ends agree, and they cannot: a SAN can be correct, a CA can be
correct, a context can be built correctly, and the handshake can still fail on hostname
verification, on a missing client certificate, or on a chain the server will not accept.
Those are the failures that reach production, because they only exist when both halves
run at once.

So this starts an actual `asyncio` TLS server configured exactly as the shim's uvicorn
is -- `tls_settings_for_this_pod()` is read here rather than restated -- and dials it
with exactly the context the control plane dials pods with. What is graded is whether
the socket comes up, and, in the negative cases, that it comes up for nothing else.

The server is a bare socket rather than a running uvicorn. What is under test is the
handshake, which happens before any HTTP is spoken, and a real uvicorn would add a
process and a port race to a case that needs neither.
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from managed_agent.control.session.pod_tls import CONTROL_PLANE_DIAL_NAME, dial_context
from managed_agent.core.tls.session_certificate import InternalCa, new_internal_ca
from managed_agent.session_shim import serve

A_POD_NAME = "map-session-11111111-1111-4111-8111-111111111111"
A_NAMESPACE = "map-dev"


@pytest.fixture
def ca() -> InternalCa:
    key, certificate = new_internal_ca()
    return InternalCa(key=key, certificate=certificate)


def _mount(directory: Path, ca: InternalCa, dns_name: str) -> None:
    """Write the three files the control plane mounts into a Session pod."""
    issued = ca.sign_for(dns_name)
    (directory / "tls.crt").write_bytes(issued.certificate_pem)
    (directory / "tls.key").write_bytes(issued.private_key_pem)
    (directory / "ca.crt").write_bytes(ca.certificate_pem)


def _pod_dns_name() -> str:
    from managed_agent.session_shim.pod_channel import shim_host

    return shim_host(A_POD_NAME, A_NAMESPACE)


@asynccontextmanager
async def _serving(
    monkeypatch: pytest.MonkeyPatch, directory: Path
) -> AsyncIterator[int]:
    """Start a socket server whose TLS is configured the way the shim's uvicorn is.

    The settings come out of `tls_settings_for_this_pod()` rather than being written out
    again, so a change to what the shim asks uvicorn for is a change to what this case
    dials -- which is the whole point of running both halves.
    """
    monkeypatch.setattr(serve, "SHIM_TLS_DIRECTORY", directory)
    monkeypatch.setattr(serve, "SHIM_CERTIFICATE_PATH", directory / "tls.crt")
    monkeypatch.setattr(serve, "SHIM_PRIVATE_KEY_PATH", directory / "tls.key")
    monkeypatch.setattr(serve, "SHIM_TRUST_BUNDLE_PATH", directory / "ca.crt")
    settings = serve.tls_settings_for_this_pod()

    context = ssl.create_default_context(
        ssl.Purpose.CLIENT_AUTH, cafile=str(settings["ssl_ca_certs"])
    )
    context.load_cert_chain(
        certfile=str(settings["ssl_certfile"]), keyfile=str(settings["ssl_keyfile"])
    )
    context.verify_mode = ssl.VerifyMode(settings["ssl_cert_reqs"])  # type: ignore[arg-type]

    async def _greet(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Read a byte, then answer one.

        The read is what forces the client certificate to be verified under TLS 1.3,
        where the client's certificate arrives after the handshake has already
        completed from the client's point of view. A server that only wrote would let a
        certificateless caller believe it was connected.
        """
        await reader.read(1)
        writer.write(b"o")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(_greet, "127.0.0.1", 0, ssl=context)
    port = server.sockets[0].getsockname()[1]
    assert isinstance(port, int)
    try:
        yield port
    finally:
        server.close()
        await server.wait_closed()


async def _dial(port: int, context: ssl.SSLContext, server_hostname: str) -> None:
    """Connect, exchange a byte, and close.

    The exchange is not decoration. Under TLS 1.3 the client finishes its handshake
    before the server has looked at its certificate, so a dial that only connected
    would report success for a caller the server is about to reject -- which is the
    exact failure `ssl_cert_reqs` exists to prevent.
    """
    reader, writer = await asyncio.open_connection(
        "127.0.0.1", port, ssl=context, server_hostname=server_hostname
    )
    writer.write(b"?")
    await writer.drain()
    assert await reader.read(1) == b"o"
    writer.close()


async def test_the_control_plane_completes_a_handshake_with_a_pod_it_signed_for(
    ca: InternalCa, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive case, and the only one that proves the other three mean anything."""
    _mount(tmp_path, ca, _pod_dns_name())
    async with _serving(monkeypatch, tmp_path) as port:
        await _dial(port, dial_context(ca), _pod_dns_name())


async def test_a_pod_signed_for_another_session_is_refused_by_hostname(
    ca: InternalCa, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name is the bound, so a certificate for another pod must not be accepted.

    Same CA, same everything else -- only the SAN differs. Without hostname verification
    this handshake succeeds, and a pod running tenant code that got hold of its own key
    could answer for every other Session at the address the control plane dials.
    """
    from managed_agent.session_shim.pod_channel import shim_host

    other = shim_host("map-session-22222222-2222-4222-8222-222222222222", A_NAMESPACE)
    _mount(tmp_path, ca, other)
    async with _serving(monkeypatch, tmp_path) as port:
        with pytest.raises(ssl.SSLCertVerificationError):
            await _dial(port, dial_context(ca), _pod_dns_name())


async def test_a_pod_signed_by_another_ca_is_refused(
    ca: InternalCa, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A certificate for the right name from the wrong CA is the forgery case."""
    other_key, other_certificate = new_internal_ca()
    _mount(
        tmp_path,
        InternalCa(key=other_key, certificate=other_certificate),
        _pod_dns_name(),
    )
    async with _serving(monkeypatch, tmp_path) as port:
        with pytest.raises(ssl.SSLCertVerificationError):
            await _dial(port, dial_context(ca), _pod_dns_name())


async def test_a_caller_with_no_certificate_of_its_own_is_refused_by_the_pod(
    ca: InternalCa, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ssl_cert_reqs` is what makes the pod's trust bundle a control.

    A server that loads a CA and then accepts connections presenting no certificate has
    one-way TLS with an unused CA file, and it looks identical in every configuration
    check. Here the client verifies the pod perfectly well and still must not get in.
    """
    _mount(tmp_path, ca, _pod_dns_name())
    anonymous = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cadata=ca.certificate_pem.decode()
    )
    async with _serving(monkeypatch, tmp_path) as port:
        with pytest.raises((ssl.SSLError, ConnectionError, AssertionError)):
            await _dial(port, anonymous, _pod_dns_name())


def test_the_control_planes_own_name_is_one_no_session_pod_can_hold() -> None:
    """Two identities the same CA signs must not be able to be confused.

    A pod's name is `<pod>.map-session.<ns>.svc.cluster.local` by construction, so a
    name outside that suffix is one no placement can produce -- which is what stops a
    leaked pod key from being usable to impersonate the control plane to another pod.
    """
    assert not CONTROL_PLANE_DIAL_NAME.endswith(".svc.cluster.local")
