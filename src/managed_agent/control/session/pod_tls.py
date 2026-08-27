"""The TLS credentials the control plane dials Session pods with.

A pod that serves TLS demands a client certificate signed by the platform CA, so the
control plane needs one of its own -- not only the CA certificate to verify the pod
with. Both halves are assembled here, once at start-up, into the `ssl.SSLContext` the
diallers in `pod_channel` carry (ADR-044).

**Why the material touches disk at all.** Python's `SSLContext.load_cert_chain` reads
files; there is no in-memory form of it, and `cryptography` cannot hand a chain to
`ssl` without one. So the certificate and key are written to a private directory this
process owns, loaded, and unlinked immediately -- the context keeps the parsed key, so
nothing has to stay on disk past this function. The window is one call inside a
directory created with mode 0700 by `mkdtemp`, which is narrower than the alternative
of leaving the material in a Secret-mounted file for the life of the pod.

**The control plane's certificate is not a pod's.** It is signed for a name no Session
pod can hold, so a leaked pod key cannot be used to impersonate the control plane to
another pod. The CA is the only thing that can mint either, and it never leaves this
process.
"""

from __future__ import annotations

import ssl
import tempfile
from pathlib import Path

from managed_agent.core.tls.session_certificate import InternalCa

CONTROL_PLANE_DIAL_NAME = "map-control-plane.internal"
"""The name the control plane's own client certificate is signed for.

Not a resolvable DNS name and deliberately so: nothing dials the control plane by it.
It exists to be *read* -- a shim logging which peer presented a certificate, an operator
reading a handshake failure -- and to be a name no Session pod's certificate can carry,
so the two identities cannot be confused for one another.
"""


def dial_context(ca: InternalCa) -> ssl.SSLContext:
    """An SSL context that verifies a Session pod and identifies this control plane.

    `check_hostname` stays on. The certificate a pod is given names exactly the address
    the diallers build, so hostname verification is the check that makes the CA mean
    something: without it, any certificate this CA ever signed -- including one held by
    a pod running tenant code -- would be accepted for any pod's address, and one
    escaped pod key would let its holder impersonate every other Session.
    """
    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH, cadata=ca.certificate_pem.decode()
    )
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    _load_our_own_certificate(context, ca)
    return context


def _load_our_own_certificate(context: ssl.SSLContext, ca: InternalCa) -> None:
    """Mint this process's client certificate and load it, leaving nothing behind.

    Written and unlinked inside one function because `load_cert_chain` takes paths and
    the loaded context does not refer back to them. A `finally` rather than a context
    manager's cleanup alone, so a load that raises still removes the private key.
    """
    issued = ca.sign_for(CONTROL_PLANE_DIAL_NAME)
    directory = Path(tempfile.mkdtemp(prefix="map-control-plane-tls-"))
    certificate = directory / "tls.crt"
    private_key = directory / "tls.key"
    try:
        certificate.write_bytes(issued.certificate_pem)
        private_key.write_bytes(issued.private_key_pem)
        private_key.chmod(0o600)
        context.load_cert_chain(certfile=certificate, keyfile=private_key)
    finally:
        certificate.unlink(missing_ok=True)
        private_key.unlink(missing_ok=True)
        directory.rmdir()
