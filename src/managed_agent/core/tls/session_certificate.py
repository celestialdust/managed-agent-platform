"""Signing a server certificate for one Session pod, and the CA that signs it.

The control plane holds a private CA and mints a certificate for each Session pod at
placement, mounting it beside the bearer token it already mounts. That is what turns the
in-cluster hops from plain HTTP into mutually authenticated TLS, so a compromised node
can no longer read a Session's token off the wire (ADR-044).

**Nothing here knows what a pod is named.** The caller passes a DNS name and gets a
certificate for exactly that name. Deriving the name here would mean this module
importing the pod-naming helper out of `control/` and the service name out of
`session_shim/`, which inverts the layering -- and it would put the one fact that must
match the URL `pod_channel` builds in a second place, free to drift from the first.

**Why a private CA at all rather than a certificate manager.** Two properties of this
platform delete most of what a manager is for. A pod is leased for one Turn (ADR-041),
so a certificate scoped to its life never renews -- there is no rotation to schedule and
no expiry to alarm on. And the material already has a delivery path: `/etc/map/shim/
token` is written per pod at creation, for that container only, so a certificate travels
the same way and nothing has to reach into a running pod. What is left is signing, which
is this file.

**The security bound is the name, not the clock.** Each pod gets its own keypair and a
certificate naming one pod, marked `CA: FALSE`. A pod runs tenant code and this
platform's sandbox is known imperfect, so the honest assumption is that a pod's private
key can escape it -- and what that must buy is nothing beyond the Session it already
belonged to. Expiry is a backstop behind that, deliberately long enough that it can
never be what ends a Turn.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Final

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CERTIFICATE_LIFETIME: Final = dt.timedelta(days=7)
"""How long a Session pod's certificate is valid for.

Long, and long on purpose. The sixty-minute Turn ceiling was removed deliberately, so
nothing bounds how long a Turn may legitimately run -- a real delegating review was
measured at forty-four minutes with its longest phases still ahead of it. A certificate
that expired mid-Turn would break the dispatch to a pod that was working, and it would
present as the platform being unreliable rather than as an expiry.

So this is sized to be irrelevant rather than tight. What actually bounds a leaked pod
key is the pod: it is destroyed when its Turn ends, and the certificate names one
Session's pod. Shortening this would buy a smaller window on a key whose reach is
already one Session, at the cost of a failure mode that fires in the middle of the
longest and most valuable runs.
"""

_CLOCK_SKEW_ALLOWANCE: Final = dt.timedelta(minutes=5)
"""How far `not_valid_before` is backdated.

A certificate stamped `not_valid_before = now` is refused by any verifier whose clock is
behind the signer's, and the failure is intermittent, node-dependent, and reads as a
network fault rather than as a clock. Backdating costs nothing that matters here,
because the certificate's reach is bounded by its name rather than by its window.
"""

_CA_LIFETIME: Final = dt.timedelta(days=3650)
"""How long a generated CA certificate is valid for.

Ten years, because this is not the rotation story. Rotating the CA means replacing the
key in the Secret and rolling the control plane, which is an operation somebody performs
deliberately -- not one an expiry date should force at whatever hour it happens to fall.
A CA that lapsed would take every Session pod with it at once.
"""

_CA_COMMON_NAME: Final = "Managed Agent Platform internal CA"

_MAX_COMMON_NAME_LENGTH: Final = 64
"""The X.509 ceiling on a common name, in characters (RFC 5280's `ub-common-name`).

Stated here because exceeding it does not degrade -- `cryptography` refuses to build the
certificate at all, at signing time, on the placement path.
"""


@dataclass(frozen=True, slots=True)
class SessionCertificate:
    """One pod's certificate and the private key that goes with it, as PEM bytes.

    PEM rather than parsed objects because both halves are on their way to a file the
    pod reads, and every TLS implementation that will load them takes PEM. Returning
    objects would mean every caller serialising them the same way, which is one more
    place for the two to disagree about encoding.

    `dns_name` is echoed back rather than left to the caller to remember. It is the one
    field a verifier actually matches on, and a caller holding a certificate without it
    would have to parse the SAN back out to know what it is good for.
    """

    dns_name: str
    certificate_pem: bytes
    private_key_pem: bytes


def new_internal_ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    """Generate a fresh CA keypair and its self-signed certificate.

    Used to create the material an operator puts into the `map-control-plane` Secret,
    and used by tests that need a CA of their own. **Not** called at startup: a control
    plane that generated a CA when it could not find one would let every replica trust a
    different root, and a restart would silently invalidate every pod already running.
    Missing material is a refusal to start, not a cue to invent some.

    P-256 rather than RSA: signing is on the placement path, the keys are smaller in the
    Secret and in the pod, and every TLS stack in this tree speaks it.
    """
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _CA_COMMON_NAME)])
    now = dt.datetime.now(dt.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _CLOCK_SKEW_ALLOWANCE)
        .not_valid_after(now + _CA_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    return key, certificate


def sign_for_a_dns_name(
    dns_name: str,
    ca_key: ec.EllipticCurvePrivateKey,
    ca_certificate: x509.Certificate,
) -> SessionCertificate:
    """A server certificate for exactly this name, and a fresh key to go with it.

    The key is generated here and never leaves with the CA's, which is what makes the
    name bound mean anything: a pod holding the CA key could sign for any name, so the
    scoping would be decoration and one escaped pod would own the platform.

    `path_length=0` on the CA above and `ca=False` here are the two halves of the same
    refusal. Without them a verifier following the chain would accept a certificate the
    pod itself signed for a name it was never given -- which is the failure that makes
    per-pod certificates worse than useless, because everything downstream would be
    trusting a chain that no longer means what it says.

    The SAN carries the name; the subject carries its **first label only**. X.509 caps a
    common name at 64 characters and a fully-qualified pod name here runs to 85, so
    putting the whole thing there is not a style choice -- it raises at signing time.
    Modern verifiers ignore the subject anyway, and one label is enough for an operator
    reading a log line to tell which pod a certificate belongs to.
    """
    common_name = dns_name.split(".", 1)[0][:_MAX_COMMON_NAME_LENGTH]
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_certificate.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _CLOCK_SKEW_ALLOWANCE)
        .not_valid_after(now + CERTIFICATE_LIFETIME + _CLOCK_SKEW_ALLOWANCE)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(dns_name)]), critical=False
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    ExtendedKeyUsageOID.SERVER_AUTH,
                    # Client auth as well, because the pod dials out too -- to the Tool
                    # Gateway and the model gateway -- and mutual authentication means
                    # the same certificate is presented in both directions. A second
                    # certificate for the outbound half would double the material in the
                    # pod to say exactly the same thing about who it is.
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    return SessionCertificate(
        dns_name=dns_name,
        certificate_pem=certificate.public_bytes(serialization.Encoding.PEM),
        private_key_pem=key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )


@dataclass(frozen=True, slots=True)
class InternalCa:
    """The signing material the control plane holds, parsed once rather than per pod.

    Bundled rather than passed as two values because the halves are only meaningful
    together: a key with the wrong certificate signs chains that verify nowhere, and the
    failure surfaces at the pod as a handshake error with nothing pointing back here.
    Parsing at construction means a malformed or mismatched pair fails at startup, which
    is a deployment that does not come up rather than one that places broken pods.

    The private key is excluded from the `repr`, for the same reason `token_key` is on
    the pod runner: one rendering of this object -- a pytest frame dump, an error
    reporter that captures locals -- would disclose the key every Session pod's identity
    is derived from. Excluded at the field so no call site has to remember.
    """

    key: ec.EllipticCurvePrivateKey = field(repr=False)
    certificate: x509.Certificate

    @classmethod
    def from_pem(cls, key_pem: bytes, certificate_pem: bytes) -> InternalCa:
        """Parse a PEM keypair, refusing a pair whose halves do not go together.

        The mismatch check is the reason this is a constructor rather than two loads at
        the call site. Two separately valid PEM files are the ordinary way this is got
        wrong -- a key rotated in the Secret without its certificate, or the reverse --
        and nothing downstream can tell: signing succeeds, the pod mounts a chain, and
        every dial fails verification against a CA certificate that did not sign it.
        """
        key = serialization.load_pem_private_key(key_pem, password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(
                "the internal CA key must be an elliptic-curve private key; "
                f"got {type(key).__name__}"
            )
        certificate = x509.load_pem_x509_certificate(certificate_pem)
        if certificate.public_key() != key.public_key():
            raise ValueError(
                "the internal CA certificate was not issued for this private key: "
                "signing with the pair would produce chains that verify nowhere"
            )
        return cls(key=key, certificate=certificate)

    @property
    def certificate_pem(self) -> bytes:
        """The CA certificate as PEM, for the trust bundle a pod is handed."""
        return self.certificate.public_bytes(serialization.Encoding.PEM)

    def sign_for(self, dns_name: str) -> SessionCertificate:
        """A server certificate for this name, signed by this CA."""
        return sign_for_a_dns_name(dns_name, self.key, self.certificate)
