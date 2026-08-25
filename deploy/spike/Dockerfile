# The spike probe's image.
#
# AL2023 matches the nodegroup's amiFamily on purpose: the RSS figures this image
# reports are read inside the container and then applied to a node running that same
# distribution and kernel, and a different libc or a different malloc would make the
# number not transfer.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# The two binaries the probe's first assertion depends on, named here because the base
# image carries neither and their absence is indistinguishable from a kernel refusal at
# exit 127.
#   util-linux -> unshare   (does this kernel allow an unprivileged user namespace)
#   bubblewrap -> bwrap     (can a sandbox actually be constructed)
# nodejs is here only to install the codex CLI below and is not used at runtime.
# The two `command -v` calls fail the build rather than the spike: an image missing
# either binary would report "the architecture does not work" about a node that works.
RUN dnf -y install util-linux bubblewrap nodejs npm tar gzip procps-ng \
 && dnf clean all \
 && command -v unshare \
 && command -v bwrap

# Pinned rather than floating. The spike's whole output is a set of numbers read off
# this binary, and a report that does not say which build produced them cannot be
# checked later. No default: a build that silently took whatever npm called `latest`
# that morning is the failure this argument exists to prevent.
ARG CODEX_VERSION
# --ignore-scripts: an npm install runs lifecycle scripts from every package in the
# transitive tree, as whatever user the build runs as, with the network up. Only the
# version of the top-level package is pinned here -- npm resolves the tree at build time,
# so a compromised transitive dependency publishing a postinstall would execute during
# `docker build` and its output would be baked into the image. The `codex --version` below
# is what makes this safe to assert rather than hope: if the CLI needed its postinstall to
# be usable, that line fails the build instead of shipping a probe that cannot run.
RUN test -n "$CODEX_VERSION" \
 && npm install -g --ignore-scripts "@openai/codex@${CODEX_VERSION}" \
 && codex --version

COPY probe.sh /opt/map/probe.sh
RUN chmod 0755 /opt/map/probe.sh

# CODEX_HOME must exist before any subcommand runs -- the CLI refuses outright rather
# than creating it -- and it must not sit under /tmp, which it refuses to write helper
# binaries into. The default ~/.codex resolves under a home directory uid 10001 does not
# own, so both are created here and handed to that uid.
ENV HOME=/home/map
ENV CODEX_HOME=/home/map/.codex
# A real group at 10001 rather than gid 0. `USER 10001` with no group runs the process in
# the root group, and the previous `chown -R 10001:0 ... chmod 0775` then gave the root
# group write access to CODEX_HOME -- so anything else on the node running as gid 0, which
# is the default for a container that names no group, could rewrite the probe's
# configuration. 0700 rather than 0750 because exactly one uid reads this tree and no group
# needs it at all.
RUN groupadd -g 10001 map \
 && useradd -u 10001 -g 10001 -d /home/map -s /sbin/nologin map \
 && mkdir -p /home/map/.codex \
 && chown -R 10001:10001 /home/map \
 && chmod -R 0700 /home/map

# Matches pod.yaml's runAsUser and runAsGroup. Built in rather than left to the manifest,
# so running this image outside Kubernetes cannot accidentally probe as root and report a
# capability an unprivileged Session pod would not have.
USER 10001:10001
ENTRYPOINT ["/opt/map/probe.sh"]
