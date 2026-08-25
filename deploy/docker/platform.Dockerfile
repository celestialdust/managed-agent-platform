# The platform services' image. One image, and all three services run it.
#
# The control plane, the Tool Gateway and the Model Gateway are one installed wheel, and
# the manifests differ only in the `command` they name. There is no per-service layer to
# put in a per-service Dockerfile, and one Dockerfile with ARG-selected branches would be
# a switch statement in a build.
#
# Two of the three ASGI factories exist as this is written.
# `managed_agent.asgi:build_app` is the control plane's and is asserted below.
# `managed_agent.composition:model_gateway_app` is the Model Gateway's and arrived with
# MAP-21. The third does not exist: `deploy/k8s/tool-gateway.yaml:34` passes uvicorn
# `composition:tool_gateway_app`, and MAP-64 adds it. What this image carries in its place
# is `tool_gateway.server:create_gateway_app`, the thing that factory will wrap, and that is what
# the assertion below names. Until MAP-64 lands, a Deployment pinned to this image
# crash-loops on uvicorn's `Attribute not found in module`, and the assertion chain cannot
# warn about it because the symbol it would import does not exist to be imported.
#
# It also carries the migration runner, which is the one thing here that is not in the
# wheel: hatchling packages src/managed_agent and nothing else, so alembic arrives on PATH
# with no revisions to run. The Job that upgrades the schema runs from this image.
#
# AL2023 matches the nodegroup and, more to the point, matches session.Dockerfile. A
# slimmer python:3.12-slim base would drop the dnf layer measured at 85.5 MB, and is
# refused because the Session image's base is fixed by the sandbox measurements
# (session.Dockerfile:8-12) and a second distribution means two CVE feeds for two images.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# Declared before every layer that reads it, so the assertion layer at the bottom sees the
# same PATH the pod will. The venv does not exist yet here; a PATH entry naming a directory
# that is not there is inert.
ENV VIRTUAL_ENV=/opt/map/venv
ENV PATH=/opt/map/venv/bin:$PATH

# No util-linux and no bubblewrap, deliberately. Only the Session pod's agent-runtime
# container runs bwrap (session-pod.yaml:121-124); these three run with
# capabilities: {drop: ["ALL"]}, so a bwrap here is a setuid-adjacent binary nothing can
# use. No nodejs and no npm either: nothing here runs codex.
#
# shadow-utils is named rather than taken transitively -- groupadd and useradd are both
# absent from the bare base image, measured, and the user layer below would fail for a
# reason that reads as a broken instruction. python3.12 because the distribution's own
# python3 is 3.9.25, also measured, and this package requires 3.12.
RUN dnf -y install \
      shadow-utils \
      python3.12 \
      python3.12-pip \
 && dnf clean all

# Every argument here is session.Dockerfile:62-92's, and every reason transfers unchanged:
# --locked so a stale lock fails the build rather than producing a dependency set nobody
# declared; --no-dev because pytest, ruff and mypy have no business in a serving image;
# --no-editable so `import managed_agent` does not depend on /build surviving, which is
# what makes deleting /build in this same layer safe; --compile-bytecode because all three
# containers run readOnlyRootFilesystem: true, so a pod that byte-compiles at import cannot
# cache the result and Python swallows the failed write rather than reporting it.
#
# The one line that is not session.Dockerfile's is the codex_cli_bin deletion, and it is
# worth its paragraph. `openai-codex` is a declared dependency because the Session pod's
# shim speaks the app-server protocol (ADR-001); it drags in openai-codex-cli-bin, which
# ships a complete codex built against musl -- measured at 258,278,208 bytes, 301 MB of a
# 508 MB venv. No platform service runs a runtime, and `import managed_agent.composition`
# loads no codex module at all (measured: the list of loaded modules matching "codex" is
# empty). So this is 123,154,024 compressed bytes -- 47% of the image, measured both ways
# -- that this image can never execute.
#
# Deleted here rather than by moving the declaration to an optional extra. That move would
# edit pyproject.toml, which has one writer by invariant I14, and uv.lock, which changes
# the Session image's digest and forces a new Environment registration. Both are shared
# with a slice that has not started. This gets the whole saving in the one image that
# cannot use the bytes, and changes no file another image reads.
#
# In the same RUN, because a deletion in a later layer removes nothing from the image.
#
# The dist-info is left in place: it records what was installed, and nothing reads the
# payload. If a release ever makes openai_codex import codex_cli_bin at import time, the
# assertion layer below fails the BUILD -- which is the point of it being down there.
COPY pyproject.toml uv.lock /build/
COPY src /build/src
RUN pip3.12 install --no-cache-dir "uv==0.9.26" \
 && uv venv --python python3.12 "$VIRTUAL_ENV" \
 && uv sync --locked --no-dev --no-editable --compile-bytecode --active --project /build \
 && pip3.12 uninstall -y uv \
 && rm -rf /build /root/.cache \
      "$VIRTUAL_ENV/lib/python3.12/site-packages/codex_cli_bin"

# The migration runner's two inputs, which the wheel does not carry.
#
# WORKDIR is load-bearing and not tidiness. alembic resolves `script_location = migrations`
# against the process's working directory, not against the directory holding alembic.ini --
# measured by running it from elsewhere, which answers
# `CommandError: Path doesn't exist: migrations.` So anything invoking alembic in this image
# runs from here, and MAP-63's Job says so with `workingDir` rather than relying on this.
COPY alembic.ini /opt/map/alembic.ini
COPY migrations /opt/map/migrations
WORKDIR /opt/map

# 10002, not the Session image's 10001. tool-gateway.yaml:26 runs runAsUser: 10002 and
# MAP-21's manifest does too; an image carrying only a 10001 passwd entry, run as 10002, has
# no entry and no HOME -- the exact failure session.Dockerfile:94-100 built its entry to
# avoid. -m creates /home/map as 10002:10002 mode 700, so no chown follows. A real group at
# 10002 rather than gid 0: USER with no group runs in the root group and runAsNonRoot checks
# the uid only.
RUN groupadd -g 10002 map \
 && useradd -m -u 10002 -g 10002 -d /home/map -s /sbin/nologin map
USER 10002:10002

# Every entry point, asserted once with the finished PATH and -- because this layer is BELOW
# the USER line -- as the uid that ships. Both halves of that placement are session.
# Dockerfile:105-112's argument and neither is symmetry: root can read and execute files uid
# 10002 cannot, so a venv that landed mode 0700 would satisfy every clause here as root and
# fail every one of them in the pod.
#
# `alembic ... heads` rather than `command -v alembic` alone: the runner on PATH with no
# revisions beside it is exactly the failure a wheel-only image produces, and heads reads the
# script directory without touching a database, so it is checkable at build time. It prints
# `0010 (head)` today.
#
# The two factories are imported, not called: build_app opens a connection pool and
# create_gateway_app builds an MCP manager, so calling either here would need a database and
# a signing key. Importing proves the module and its dependency tree are installed, which is
# the thing a --no-dev install gets wrong.
#
# `import managed_agent.composition` rather than `import managed_agent`: the package's
# __init__.py is empty, so importing it loads no third-party module and the check cannot fail
# for a missing dependency. composition is the widest import in the tree -- sqlalchemy,
# aioboto3, fastapi, pydantic -- so it is the clause that fails if a runtime dependency was
# declared dev-only, and it is also what would fail if the codex_cli_bin deletion above ever
# stopped being safe.
#
# The three absences are asserted because each is a thing this image is defined by not having.
# All three are absent from the bare base image too (measured), so none of them can pass by
# accident: bwrap and unshare arrive only with bubblewrap and util-linux, and codex only with
# the npm layer this file does not have.
RUN command -v uvicorn \
 && command -v alembic \
 && alembic -c /opt/map/alembic.ini heads | grep -q '(head)' \
 && python -c "from managed_agent.asgi import build_app; assert callable(build_app)" \
 && python -c "from managed_agent.gateway.tool.server import create_gateway_app; assert callable(create_gateway_app)" \
 && python -c "import managed_agent.composition" \
 && ! command -v codex \
 && ! command -v bwrap \
 && ! command -v unshare

# No ENTRYPOINT and no CMD of its own, for session.Dockerfile:154-168's reason: three
# Deployments run this image with three different `command`s, so a default would never be
# used in the cluster -- and outside it, a default that was one of the three would present a
# process that cannot work as the image's purpose. The base image's CMD ["/bin/bash"] is
# inherited, so a bare `docker run` opens a shell as uid 10002 rather than refusing; omitting
# a default keeps this image from asserting a role it cannot fulfil and does nothing more.
