"""What the control plane signs for a Session pod, and what it refuses to sign.

Tier 1, no infrastructure. A certificate is only worth anything for what it *refuses*,
so nearly every case here is a negative: a name it does not cover, an issuer it was not
signed by, a lifetime that has run out. A test that only asserted "a certificate came
back" would pass against one signed by nobody for nothing.

The property that makes this design small is ADR-041's: a pod is leased for one Turn and
its DNS name is a pure function of its Session. So the name is known before the pod
exists, the certificate is scoped to it, and there is no renewal path to get wrong --
which is why the lifetime here is a backstop rather than the control. The control is
that a leaked pod key reaches exactly one Session's name, and the case at the bottom is
what says so.
"""

from __future__ import annotations

import datetime as dt

import pytest
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec

from managed_agent.control.session.placement import pod_name_for
from managed_agent.core.ids import new_session_id
from managed_agent.core.tls.session_certificate import (
    CERTIFICATE_LIFETIME,
    SessionCertificate,
    new_internal_ca,
    sign_for_a_dns_name,
)
from managed_agent.session_shim.serve import SHIM_SERVICE

_NAMESPACE = "map-dev"


def _a_pods_name() -> str:
    """The name `pod_channel` really dials, built the way production builds it.

    Assembled from the same two helpers rather than from a literal, so a case here fails
    if either half moves -- which is the failure that would otherwise ship a certificate
    for a name nothing resolves.
    """
    pod = pod_name_for(new_session_id())
    return f"{pod}.{SHIM_SERVICE}.{_NAMESPACE}.svc.cluster.local"


@pytest.fixture
def ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    return new_internal_ca()


def _ec_public_key(certificate: x509.Certificate) -> ec.EllipticCurvePublicKey:
    """The certificate's public key, narrowed to the curve this module signs on.

    `Certificate.public_key()` is typed as the union of every algorithm X.509
    admits, and the two methods used below -- `verify` and `public_numbers` -- do
    not exist on all of them. Narrowing here rather than silencing at each call
    site means a switch away from EC fails this assertion with a readable
    message instead of an attribute error.
    """
    key = certificate.public_key()
    assert isinstance(key, ec.EllipticCurvePublicKey)
    return key


def _issued(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> tuple[SessionCertificate, x509.Certificate]:
    key, cert = ca
    issued = sign_for_a_dns_name(_a_pods_name(), key, cert)
    return issued, x509.load_pem_x509_certificate(issued.certificate_pem)


def test_the_certificate_covers_the_name_the_control_plane_will_dial(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """The SAN is the pod's real in-cluster name, not its bare pod name.

    TLS verification matches the name in the URL, and the URL `pod_channel` builds is
    the fully-qualified `<pod>.<service>.<ns>.svc.cluster.local`. A certificate carrying
    only the short name would verify in no client and would fail every dispatch -- while
    looking correct in any test that read the subject instead of the SAN.
    """
    issued, parsed = _issued(ca)

    names = parsed.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value.get_values_for_type(x509.DNSName)
    assert names == [issued.dns_name]
    assert issued.dns_name.endswith(f".{_NAMESPACE}.svc.cluster.local")


def test_two_sessions_get_certificates_that_do_not_cover_each_other(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """The bound on a leaked pod key, which is the whole security argument.

    A pod runs tenant code and this platform's sandbox is known imperfect, so the honest
    assumption is that a pod's private key can escape it. What that must not buy is any
    other Session -- so the names must be disjoint, and asserting the two certificates
    merely differ would not show it.
    """
    first, first_parsed = _issued(ca)
    second, second_parsed = _issued(ca)

    def names(parsed: x509.Certificate) -> list[str]:
        return parsed.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.DNSName)

    assert first.dns_name != second.dns_name
    assert not set(names(first_parsed)) & set(names(second_parsed))


def test_the_certificate_is_signed_by_the_ca_that_issued_it(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """Verified against the CA's public key rather than assumed from the issuer name.

    An issuer *name* is a string anybody can write. The signature is the only thing that
    makes this certificate mean anything, so it is checked with the CA's own key -- and
    checked again below against a different CA, because a verification that passes for
    every key verifies nothing.
    """
    _, ca_cert = ca
    _, parsed = _issued(ca)

    algorithm = parsed.signature_hash_algorithm
    assert algorithm is not None
    _ec_public_key(ca_cert).verify(
        parsed.signature, parsed.tbs_certificate_bytes, ec.ECDSA(algorithm)
    )
    assert parsed.issuer == ca_cert.subject


def test_a_certificate_from_one_ca_does_not_verify_against_another(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """The negative half, without which the case above proves nothing."""
    _, parsed = _issued(ca)
    _, other_ca_cert = new_internal_ca()

    algorithm = parsed.signature_hash_algorithm
    assert algorithm is not None
    with pytest.raises(InvalidSignature):
        _ec_public_key(other_ca_cert).verify(
            parsed.signature, parsed.tbs_certificate_bytes, ec.ECDSA(algorithm)
        )


def test_the_lifetime_outlasts_any_turn_that_could_still_be_running(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """Expiry is a backstop here, and it must never be what ends a Turn.

    The sixty-minute Turn ceiling was removed on purpose, so there is no bound on how
    long a Turn may legitimately run -- a real delegating review was measured at
    forty-four minutes and its longest phases came after that. A certificate that
    expired mid-Turn would break the dispatch to a pod that was working, and it would
    present as the platform being unreliable rather than as an expiry.

    So the lifetime is sized to be irrelevant. What actually bounds a leaked pod key is
    the pod: it is destroyed when its Turn ends, and the key reaches one Session's name.
    """
    assert dt.timedelta(days=1) <= CERTIFICATE_LIFETIME

    _, parsed = _issued(ca)
    granted = parsed.not_valid_after_utc - parsed.not_valid_before_utc
    assert granted >= CERTIFICATE_LIFETIME


def test_the_certificate_is_valid_now_and_tolerates_a_skewed_clock(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """`not_valid_before` is backdated, because nodes disagree about the time.

    A certificate stamped `not_valid_before = now` is refused by any verifier whose
    clock is a second behind the signer's, and the failure is intermittent, node-
    dependent and reads as a network fault. Backdating costs nothing here: the
    certificate is already scoped to one pod's name.
    """
    now = dt.datetime.now(dt.UTC)
    _, parsed = _issued(ca)

    assert parsed.not_valid_before_utc < now
    assert parsed.not_valid_after_utc > now


def test_the_pod_key_is_not_the_ca_key(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """Each pod gets its own keypair, which is what makes the SAN bound mean anything.

    Handing every pod the CA's key would make each one able to sign a certificate for
    any name -- so the careful scoping above would be decoration, and one escaped pod
    would own the whole platform.
    """
    ca_key, _ = ca
    issued, parsed = _issued(ca)

    assert issued.private_key_pem not in (
        ca_key.private_numbers().private_value.to_bytes(32, "big"),
    )
    assert (
        _ec_public_key(parsed).public_numbers() != ca_key.public_key().public_numbers()
    )


def test_a_session_pod_certificate_cannot_sign_anything(
    ca: tuple[ec.EllipticCurvePrivateKey, x509.Certificate],
) -> None:
    """`CA: FALSE`, so an escaped pod key cannot mint a certificate for another Session.

    Without the constraint the SAN scoping is advisory: any verifier following the chain
    would accept a certificate the pod signed for a name it was never given. This is the
    one extension whose absence is silently fatal.
    """
    _, parsed = _issued(ca)

    basic = parsed.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert basic.ca is False
