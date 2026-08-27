# The Session pod's image. One image, and all three containers in the pod run it.
#
# Why one image rather than two: the pod needs `codex` in one container and the
# managed_agent package in another, and there is exactly one ECR repository for the
# Session pod (map/session-shim). A pod pulling two images would need a repository this
# platform does not have. So this carries both halves and the pod picks with `command`.
#
# AL2023 matches the nodegroup: `aws eks describe-nodegroup` reports amiType
# AL2023_x86_64_STANDARD on t3.medium. That is not tidiness -- every filesystem-deny
# finding the sandbox rests on was measured against this distribution's bubblewrap and
# util-linux, and a different distribution's bwrap is a different measurement of the one
# boundary the architecture cannot afford to re-take blind.
FROM public.ecr.aws/amazonlinux/amazonlinux:2023

# No default, deliberately. A build that quietly took whatever npm called `latest` that
# morning produces a pod whose runtime version nothing in the tree records, and the
# sandbox findings this platform rests on are readings off one specific build. The name
# and the no-default are also a contract: the push script reads them from outside.
ARG CODEX_VERSION

# Declared before every layer that reads it, so the npm layer and the assertion layer at
# the bottom see the same PATH the pod will. The venv does not exist yet at this point;
# a PATH entry naming a directory that is not there yet is inert.
ENV VIRTUAL_ENV=/opt/map/venv
ENV PATH=/opt/map/venv/bin:$PATH

# util-linux -> unshare and bubblewrap -> bwrap are the sandbox. The runtime compiles
# every confined command into a bwrap argv, so an image without them does not run a
# weaker sandbox, it runs none. Both are proven resolvable here, when the image is
# built: a missing binary exits 127, which is the same non-zero status a kernel refusing
# user namespaces produces, and read off the exit status alone the wrong one of those two
# failures says the node cannot host a sandbox at all.
#
# shadow-utils is named rather than taken transitively: `groupadd` and `useradd` are both
# absent from the bare base image and exit 127 there, and the user layer below would then
# fail for a reason that reads as a broken instruction rather than a missing package.
#
# python3.12 because the distribution's default python3 is 3.9 and this package requires
# 3.12. npm exists only to install the pinned codex below; nothing at runtime uses it.
#
# poppler-utils and qpdf are the CONFINED AGENT's tools, not the platform's -- nothing
# in this repository shells out to either. They are the command-line programs that
# Anthropic's published `pdf` skill tells the model to run: `pdftotext` and `pdfimages`
# come from poppler-utils, and the skill's own summary table names `qpdf` as its
# command-line merge tool. A skill whose instructions cannot be followed fails as
# though the model were at fault, so what it names is installed here rather than left
# to a Turn to fetch.
#
# Two programs that skill names are deliberately NOT here, and both absences were
# measured against this base image rather than assumed:
#
#   tesseract -- not packaged for AL2023 under any name. It backs the skill's OCR
#                branch, which therefore cannot run in this image. The pytesseract
#                wrapper IS installed, so a model following that branch reaches a
#                missing binary rather than a missing Python module -- one step later,
#                with a message that points at the real gap.
#   pdftk     -- not packaged either, and optional in the skill's own text: its heading
#                reads "pdftk (if available)" and the summary table routes command-line
#                merging to qpdf. Adding it would mean a JRE (corretto-17 is available)
#                plus an unpackaged jar, to duplicate what qpdf already does.
RUN dnf -y install \
      util-linux \
      bubblewrap \
      shadow-utils \
      nodejs \
      npm \
      python3.12 \
      python3.12-pip \
      poppler-utils \
      qpdf \
 && dnf clean all \
 && command -v unshare \
 && command -v bwrap

# --ignore-scripts: an npm install runs lifecycle scripts from every package in the
# transitive tree, as the build user, with the network up. Only the top-level version is
# pinned here -- npm resolves the rest at build time -- so a compromised transitive
# dependency publishing a postinstall would execute during `docker build` and be baked
# in. The version assertion at the bottom is what makes that safe to state rather than
# hope: if the CLI needed its postinstall to be usable, the build fails there.
RUN test -n "$CODEX_VERSION" \
 && npm install -g --ignore-scripts "@openai/codex@${CODEX_VERSION}" \
 && npm cache clean --force

# uv rather than a bare `pip install .`: uv.lock is committed and is what decides which
# versions install, so a plain pip install would resolve the `>=` bounds afresh on every
# build and two images built a week apart would carry different fastapi. uv is installed
# with the pip the distribution already ships rather than pulled from a second registry,
# and removed in the same layer -- the pod runs model-driven code and has no business
# carrying a tool that resolves and installs arbitrary packages.
#
# --locked, not --frozen: --frozen installs what the lock says, --locked additionally
# fails the build if uv.lock has fallen out of step with pyproject.toml. An image built
# from a stale lock has a dependency set nobody declared.
#
# --no-dev: the dev extra is pytest, ruff, mypy, testcontainers and type stubs. pyyaml
# and kubernetes-asyncio are NOT in it -- they are runtime dependencies -- because a
# dev-extra declaration of a module src/ imports produces an image that fails at import
# rather than at build time.
#
# --no-editable installs a wheel, so `import managed_agent` does not depend on /build
# surviving -- which is what makes deleting /build in this same layer safe, leaving one
# copy of the source in the image instead of two.
#
# `--extra agent-tools` installs reportlab and pypdf, which are the CONFINED AGENT's
# tools and not this package's dependencies -- nothing under `src/` imports either. They
# go into this venv rather than the system interpreter because the image's PATH puts the
# venv first, so the agent's bare `python3` is this one: a `pip3.12 install` beside it
# is invisible to every command the agent runs, which the build assertion at the bottom
# measured before this line said `--extra`.
#
# Locked rather than resolved at build time, and installed rather than left to a Turn
# even though an Environment can now grant egress to pypi.org. An Environment granting
# no domain is the default, a skill that only works with egress is a skill failing for a
# reason nowhere in its own text, PyPI being down would become a failed Turn a tenant
# paid for, and a version resolved per Turn means two Sessions in one Environment run
# different code -- which is the guarantee `core/environment.py` exists to make.
#
# --compile-bytecode: every container here runs with readOnlyRootFilesystem: true, so a
# pod that has to byte-compile at import cannot cache the result and pays it again on the
# next start, and Python swallows the failed write rather than reporting it. Measured:
# 1762 files, 949 ms, once, at build time.
COPY pyproject.toml uv.lock /build/
COPY src /build/src
RUN pip3.12 install --no-cache-dir "uv==0.9.26" \
 && uv venv --python python3.12 "$VIRTUAL_ENV" \
 && uv sync --locked --no-dev --extra agent-tools --no-editable --compile-bytecode \
      --active --project /build \
 && uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache "pip==26.0" \
 && pip3.12 uninstall -y uv \
 && rm -rf /build /root/.cache

# The model catalogue the Agent Runtime validates a spawned agent's model against.
#
# Why this is here at all: the catalogue compiled into the codex binary names eight
# OpenAI models and nothing else, so every attempt to delegate to a model this platform
# serves is refused. The agent tries five times, tells the tenant it cannot, and does the
# work itself -- a Turn that succeeds while the feature a tenant enabled does nothing.
#
# Why at build time rather than per Session. `model_catalog_json` takes a filesystem
# path, and the document is 234 KB. A Session's requirements Secret has roughly 200 KiB
# left after skill delivery, so there is no number of models at which a per-Session copy
# fits -- and a Secret over its ceiling is not truncated, the pod simply never schedules.
# Written once into the image it costs a Session nothing.
#
# Why extracted from the binary rather than checked into the repository. A vendored copy
# can describe a different runtime than the one shipped and nothing about the file would
# say so; the symptom is a spawn refused for a model the catalogue names. Reading it out
# of the binary installed above makes that state unrepresentable rather than guarded.
#
# `/opt/codex` and not `/etc/codex`: the pod mounts the requirements Secret over
# `/etc/codex`, which would hide anything the image put there. Nothing mounts over /opt.
#
# The binary is resolved by searching for the catalogue rather than by trusting `codex`
# on PATH. npm's launcher carries no catalogue, and the self-check below -- which refuses
# a codex resolving inside the venv -- runs after this, so a bake trusting PATH could
# extract from one runtime while the pod executes another.
COPY tools/bake_model_catalog.py /bake/bake_model_catalog.py
COPY deploy/k8s/model-gateway.yaml /bake/model-gateway.yaml
#
# `npm root -g` rather than a hardcoded `/usr/lib/node_modules`: the global prefix is
# npm's to choose, and a wrong guess here would fail this build for a reason that reads
# like a missing runtime. npm is still installed at this point -- it is never removed,
# only its cache is cleaned.
RUN NODE_MODULES="$(npm root -g)" \
 && test -d "$NODE_MODULES/@openai" \
 && "$VIRTUAL_ENV/bin/python" /bake/bake_model_catalog.py \
      --binary "$NODE_MODULES/@openai" \
      --routing-table /bake/model-gateway.yaml \
      --out /opt/codex/models.json \
 && chmod 0444 /opt/codex/models.json \
 && rm -rf /bake

# pip, installed AFTER the sync and never by `uv venv --seed`. That ordering is the
# whole of this line. `uv sync` prunes whatever the lock does not name, so a seeded
# pip is installed and then deleted again in the same layer -- and the failure is a
# `map-pip` reporting "No module named pip" on a Turn, not at build time.
#
# Why the confined agent needs a pip at all. Anthropic's published `pdf` skill opens a
# recipe with `pip install pytesseract pdf2image`, and a skill is a document written for
# a laptop. Measured inside the sandbox on a live pod: there was no pip on the agent's
# PATH at all, so that line failed with `command not found` -- a message that names
# nothing a reader could act on. The six distributions that skill needs are baked in
# above, so its own setup step is unnecessary; this is for the NEXT skill, whose
# dependencies nobody here could have anticipated.
#
# Pinned like everything else in this image. pip is not in uv.lock -- it is not a
# dependency of this package, it is a tool the agent runs -- so nothing else would pin
# it, and an unpinned install is a different pip in every image built.

# The one command the agent runs to install a package, and the reason it is a wrapper.
#
# A plain `pip install` cannot work here and the four reasons are all measured, none of
# them mentioned by any skill's own text: the venv it would install into is a read-only
# filesystem, pip's scratch directory would be on a read-only /tmp (that mount exists
# for the bubblewrap helper OUTSIDE the sandbox), the result has to be on PYTHONPATH to
# import, and the only writable path a confined command has is its workspace. Telling
# the model all four is a paragraph it can get wrong on any Turn. Telling it one command
# name is a paragraph it cannot.
#
# On pod-local scratch and no longer under the workspace (ADR-037). It sat at
# <workspace>/.map/lib while the workspace was pod-local, under a dotted directory so
# `_is_a_bare_leaf` in shim/serve.py would exclude the tree from ship-out. The workspace
# is a network mount now, so leaving it there would put every run-time install across
# NFS for bytes that are rebuildable by definition -- and the dot is no longer what
# keeps a site-packages tree out of a tenant's deliverables: ship-out walks the
# workspace and this path is not under it.
#
# `core/pod/workspace_contract.py` owns both names and is what tells the model this
# command exists; the paths here and there are compared by
# tests/deploy/test_the_image_honours_the_workspace_contract.py, because a wrapper
# writing somewhere the contract does not name is a promise the platform breaks.
#
# TMPDIR is still set inline for the one command that needs it, rather than being an
# image-wide default. The image sets no TMPDIR at all and that is deliberate -- see the
# ENV block at the bottom of this file.
#
# `exec` so the agent sees pip's own exit status and pip's own error text. A wrapper
# swallowing either would turn "no such package" into a silent failure inside a Turn.
#
# `mkdir -p` on the target directory first, because this base image does not have one:
# the first build of this layer failed with `/usr/local/bin/map-pip: No such file or
# directory`, which reads as a broken redirect rather than a missing parent.
#
# A heredoc rather than `printf '%s\n'` with a line per argument. The wrapper's own body
# contains backslash continuations, and in the printf form each of those needs escaping
# against the Dockerfile parser as well -- so the text a reader sees is not the text the
# file gets, on precisely the lines where that difference would break it.
RUN mkdir -p /usr/local/bin \
 && cat > /usr/local/bin/map-pip <<'WRAPPER' \
 && chmod 0755 /usr/local/bin/map-pip \
 && sh -n /usr/local/bin/map-pip
#!/bin/sh
# Install a Python package where this Session can import from.
# Written by deploy/docker/session.Dockerfile; see core/workspace_contract.py.
set -eu
lib="/session/scratch/lib"
tmp="/session/scratch/tmp"
mkdir -p "$lib" "$tmp"
TMPDIR="$tmp" exec python3 -m pip install \
  --no-cache-dir --no-input --disable-pip-version-check \
  --target "$lib" --upgrade "$@"
WRAPPER

# Matches the pod's runAsUser and runAsGroup. Built in rather than left to the manifest,
# so running this image outside Kubernetes cannot accidentally run as root. -m creates
# /home/map explicitly instead of depending on login.defs CREATE_HOME; it lands as
# 10001:10001 mode 700, so no chown or chmod is needed after it. A real group at 10001
# rather than gid 0: USER with no group runs the process in the root group, and
# runAsNonRoot checks the uid only. A passwd entry also means the numeric USER below
# still resolves HOME=/home/map, which the assertion layer runs with.
RUN groupadd -g 10001 map \
 && useradd -m -u 10001 -g 10001 -d /home/map -s /sbin/nologin map
USER 10001:10001

# Every half, re-asserted once with the finished PATH -- and, because this layer is
# BELOW the USER line, as the uid that ships rather than as root. Both halves of that
# placement are load-bearing and neither is symmetry.
#
# After USER: root can read and execute files uid 10001 cannot. A venv or a node_modules
# tree that landed mode 0700 would satisfy every clause here under root and fail every
# one of them in the pod, with the build green and nothing in the tree saying why. An
# image can contain a binary the shipping user cannot run.
#
# After the venv: a `command -v codex` inside the layer that installs codex runs before
# the venv exists, so it cannot see a shadow even in principle, and a check that cannot
# fail is not a check.
#
# The case arm is the one that needs explaining. `openai-codex`, which pyproject.toml
# declares, depends on openai-codex-cli-bin, which ships its own complete codex -- a
# different version (0.147.0), built against musl, sitting inside the venv, 258 MB of it.
# Nothing in this repository imports either package. Today neither distribution installs
# a console script, so that copy is not on PATH and this arm cannot fire; it fires if a
# future release adds one, or a lock refresh moves the bundled version. Asserting the
# *version* alone would not catch either if that copy's version had caught up; asserting
# *which* codex resolves does.
#
# `codex app-server`, not just `codex`: the pod's command is `codex app-server --listen
# unix://…`, and an image where `codex --version` answers but the subcommand does not is
# an image that satisfies a version check and cannot start the pod.
#
# `import managed_agent.composition`, not `import managed_agent`: the package's
# __init__.py is empty, so importing it loads no third-party module and the check cannot
# fail for a missing dependency. composition is the widest import in the package -- it
# pulls sqlalchemy, aioboto3, fastapi, pydantic -- so it is the clause that fails if a
# runtime dependency was declared dev-only. No process in this pod imports it; it is
# here as the proof that `uv sync --no-dev` installed the declared set.
#
# The agent's own PDF toolchain is exercised rather than imported, and the difference is
# the whole reason those clauses are long. An import proves a distribution is on the
# agent's `sys.path`; it does not prove the thing works. reportlab needs a C-compiled
# freetype to render text, pdfplumber needs pdfminer's own parser to agree with what
# reportlab wrote, and `pdftotext` is a binary that exits 127 when poppler-utils is
# absent -- three failures an import list is blind to. So this WRITES a PDF, pulls its
# text back out with poppler, merges it with qpdf, counts the pages with pypdf, and
# re-reads the text with pdfplumber: one round trip through every tool that Anthropic's
# `pdf` skill names and this platform ships, on the pod's own PATH, as the shipping uid.
#
# `pytesseract` is imported and NOT exercised, and that is not an oversight. It is a
# wrapper around the `tesseract` binary, which AL2023 does not package -- see the dnf
# layer above. An assertion here would fail the build for a gap the dnf layer already
# records in words, and dropping the import clause would let the distribution silently
# disappear from the lock.
#
# The shim's entry point is imported rather than run. `build_shim_app` is a factory that
# opens the runtime connection in its lifespan, not at import, so importing it here needs
# no socket -- and an image whose shim module is missing or unimportable fails the build
# instead of failing every readiness probe in the cluster.
RUN command -v unshare \
 && command -v bwrap \
 && codex --version | grep -qF "codex-cli ${CODEX_VERSION}" \
 && case "$(command -v codex)" in \
      "$VIRTUAL_ENV"/*) \
        echo "codex resolves inside the venv, not the pinned install" >&2; exit 1 ;; \
    esac \
 && codex app-server --help >/dev/null \
 && test -r /opt/codex/models.json \
 && "$VIRTUAL_ENV/bin/python" -c "import json; d = json.load(open('/opt/codex/models.json')); assert [m for m in d['models'] if m['visibility'] == 'list'], 'the baked catalogue offers no model, so every spawn would be refused'" \
 && python -c "import managed_agent.composition" \
 && command -v uvicorn \
 && python -c "from managed_agent.session_shim.serve import build_shim_app; assert callable(build_shim_app)" \
 && python3 -c "import pandas, pdf2image, pdfplumber, pypdf, pytesseract, reportlab" \
 && python3 -m pip --version \
 && command -v map-pip \
 && test -x /usr/local/bin/map-pip \
 && command -v pdftotext \
 && command -v pdfimages \
 && command -v qpdf \
 && python3 -c "from reportlab.pdfgen.canvas import Canvas; \
page = Canvas('/tmp/probe.pdf'); page.drawString(72, 720, 'probe'); page.save()" \
 && head -c 5 /tmp/probe.pdf | grep -qF '%PDF-' \
 && pdftotext /tmp/probe.pdf /tmp/probe.txt \
 && grep -qF probe /tmp/probe.txt \
 && qpdf --empty --pages /tmp/probe.pdf /tmp/probe.pdf -- /tmp/merged.pdf \
 && python3 -c "from pypdf import PdfReader; \
assert len(PdfReader('/tmp/merged.pdf').pages) == 2" \
 && python3 -c "import pdfplumber; \
assert 'probe' in pdfplumber.open('/tmp/probe.pdf').pages[0].extract_text()" \
 && rm -f /tmp/probe.pdf /tmp/probe.txt /tmp/merged.pdf

# Where a build tool's output and caches go, which is pod-local scratch and not the
# workspace (ADR-037).
#
# A tool decides these paths and no instruction to the model reaches them: `cargo build`
# writes to target/, `npm install` to its cache, pip and uv to theirs, and the agent did
# not choose any of them. Under ADR-035 the workspace is a network mount, so left alone
# every one of those writes is a round trip for bytes nobody needs afterwards. These
# variables are the half of ADR-037 that needs no compliance from the agent.
#
# BELOW every build layer on purpose. npm, pip and uv all run above this line, and a
# cache variable declared before them would point the build itself at a directory that
# exists only inside a Session pod.
#
# In the image rather than on the container, unlike PYTHONPATH beside them in
# session-pod.yaml, and the difference is which processes may see the value. PYTHONPATH
# has to be scoped: set here it would also be on the shim's own interpreter, putting a
# tenant's installed packages on the import path of the process that serves this pod's
# HTTP. These name tools nothing in this pod runs outside the sandbox, so the image is
# where a default belonging to a tool belongs.
#
# Most of them name a toolchain this image does not carry -- there is no cargo, no rustc
# and no go here, and network egress is off unless an Environment granted it, so nothing
# can fetch one either. They are set anyway, and cheaply: a toolchain that arrives later,
# in a derived image or through a Turn, then lands pod-local by default instead of
# discovering the mount. `npm_config_cache` and `PIP_CACHE_DIR` are the two with a tool
# present today.
#
# THERE IS DELIBERATELY NO TMPDIR HERE, and ADR-037 asks for one. All three containers
# run this image, so an `ENV TMPDIR=` reaches the Agent Runtime's own process
# environment, and `std::env::temp_dir()` in that process is where the sandbox helper
# builds its registry of synthetic bwrap mount targets before every confined command --
# outside the sandbox. Pointing it at /session/scratch would hand the confined agent
# write access to the staging area its own confinement is assembled in, and pointing it
# anywhere else makes the /tmp mount in session-pod.yaml inert while it still reads as
# present, which was measured on map-dev as worse than the panic it was meant to avoid.
# The variable cannot be aimed at the agent without also aiming it at the runtime,
# because they are one process's environment. Not setting it changes nothing ADR-037
# was protecting: with TMPDIR unset a tool's temporary files go to /tmp, which bwrap
# ro-binds, so they never reached the mount either. What is missing is a writable
# temporary directory for the agent, which the contract's scratch clause names instead.
# `test_no_container_redirects_the_system_temporary_directory` refuses it in this file
# as well as in the manifest.
ENV CARGO_TARGET_DIR=/session/scratch/cargo-target
ENV npm_config_cache=/session/scratch/npm
ENV PIP_CACHE_DIR=/session/scratch/pip
ENV UV_CACHE_DIR=/session/scratch/uv
ENV GOCACHE=/session/scratch/go-build

# No ENTRYPOINT and no CMD *of its own*. Three containers run this image with three
# different commands and all three name `command` explicitly, so a default here is never
# used in the pod -- and outside it, a default that was one of the three would start a
# shim with no runtime beside it and no Session id, presenting a process that cannot work
# as the image's purpose.
#
# That is the whole reason, and it is worth being exact about what omitting them does
# *not* buy, because an earlier version of this comment claimed it and was wrong. The
# base image's `CMD ["/bin/bash"]` is **inherited**, so a bare `docker run` on this image
# starts an interactive shell as uid 10001 rather than refusing for want of a command:
#
#   docker inspect --format '{{.Config.Entrypoint}} {{.Config.Cmd}}'  ->  [] [/bin/bash]
#
# Omitting a default keeps this image from *asserting* a role it cannot fulfil. It does
# not make running it outside Kubernetes inert, and nothing here tries to.
