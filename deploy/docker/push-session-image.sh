#!/bin/sh
# Build the Session image and put it in the registry a cluster node pulls from.
#
# A node pulls from a registry and never from a developer's Docker daemon, so an image
# that exists only locally cannot start a Session pod. This is the command that closes
# that gap, and it is a command a person runs: CI checks this repository and does not
# deploy it, because the account it pushes to is shared and the credentials that reach
# it should stay with an operator. `make image-session` is this script.
#
# The tag names the two inputs the tree controls -- the commit, and the runtime version
# the image's ARG demands -- and nothing else. It is not the image's identity: a pod's
# image comes from `Environment.runtime_image`, which refuses anything that is not
# `name@sha256:<64 hex>`, so the bytes are named by the digest this prints at the end.
# The tag exists so that a person holding a digest can find out what built it.
#
# The repository is IMMUTABLE, which makes the tag a name for one byte set and a second
# push under it a refusal rather than an overwrite. That is the point. `map/spike` is
# MUTABLE and the same tag was pushed to it twice on 2026-08-22: `docs/progress.md`
# records `0.149.0` resolving to two different digests, and only the later one still
# carries the tag. Thirty-nine of that repository's forty-one manifests are orphans.
#
# It refuses to run while a build input is uncommitted, because the tag would then name
# a commit that is not what was built. Only the inputs count: the build context admits
# `pyproject.toml`, `uv.lock` and `src/**`, and BuildKit reads the two files named
# below, so an uncommitted document elsewhere in the tree changes no byte of the image
# and is not grounds for refusing.
set -eu

REPOSITORY=map/session-shim
PLATFORM=linux/amd64
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

: "${CODEX_VERSION:?no default; the image ARG refuses one too}"

# Before the print-tag seam below, so the seam cannot be used to get a tag for a tree
# that would not be allowed to push one.
#
# The pathspecs are written out rather than held in a variable: an unquoted expansion is
# what would make one word into several, and shellcheck is right to refuse that.
uncommitted=$(git -C "$ROOT" status --porcelain -- \
  pyproject.toml uv.lock src \
  deploy/docker/session.Dockerfile deploy/docker/session.Dockerfile.dockerignore)
if [ -n "$uncommitted" ]; then
  echo "a build input is uncommitted, so the tag would name a commit that is not" >&2
  echo "what was built. Commit or stash these and run again:" >&2
  echo "$uncommitted" >&2
  exit 1
fi

TAG="git-$(git -C "$ROOT" rev-parse HEAD)-codex-$CODEX_VERSION"

# The seam that makes the tag rule checkable with no daemon, no account and no push.
if [ "${MAP_PRINT_TAG_ONLY-}" = 1 ]; then
  printf '%s\n' "$TAG"
  exit 0
fi

# Derived rather than written down, so this file carries no account id and no region:
# the profile decides both already, and a literal here would be a second answer free to
# disagree with the one every other call in this repository resolves.
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

# An already-present tag means the bytes for this commit and this runtime version are
# in the registry, so there is nothing to push: read the digest and stop. Checked
# before the build, and it still prints the digest, because the digest is the whole
# output anybody uses.
#
# What it saves is measured rather than assumed. Re-pushing the identical manifest
# under an existing tag is *accepted* by ECR -- measured 2026-08-22, exit 0 -- so this
# does not guard against a crash on a warm cache; it skips a build and a push that
# change nothing. The refusal it does avoid is the rebuild that differs: the base image
# is a mutable tag and both `dnf` and `npm install -g` resolve fresh, so one commit can
# produce two byte sets, and a different digest under a claimed tag is what an
# IMMUTABLE repository refuses -- after the build has already run.
present=$(aws ecr describe-images --repository-name "$REPOSITORY" \
  --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' \
  --output text 2>/dev/null || true)
if [ -n "$present" ] && [ "$present" != "None" ]; then
  echo "$TAG is already in $REPOSITORY, and the repository is IMMUTABLE: these are" >&2
  echo "the bytes this commit and this runtime version already produced." >&2
  printf '%s/%s@%s\n' "$REGISTRY" "$REPOSITORY" "$present"
  exit 0
fi

aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

# No attestation, built into the local image store, and pushed as one separate command.
# All three parts are one decision: exactly one manifest is PUT under the tag.
#
# Measured against this repository on 2026-08-22, twice, because the first reading was
# misdiagnosed. By default buildx exports three objects for a one-platform build -- the
# image manifest, a provenance attestation manifest, and an index over both -- and it
# binds the tag to the index. ECR then refuses one of the two children under that tag
# with a bare `400 Bad Request`: the platform manifest when the index already holds the
# tag, the attestation manifest when it does not. Either way the script exits non-zero
# after the image has already reached the registry, which is the worst outcome there
# is: the next person pushes again, at a new commit, for nothing.
#
# `--provenance=false` removes the attestation, and removes the index with it, so the
# export is one plain manifest and there is one thing to PUT. What is given up is
# buildx's provenance record, which nothing in this tree reads and which the tag already
# answers -- it names the commit and the runtime version outright.
#
# `--load` and a separate `docker push` rather than `--push`, because under Docker
# Desktop's `docker` builder driver `--push` pushes from buildkit AND from the daemon
# copy the same build loaded: two PUTs of a tag that allows one.
docker buildx build \
  --platform "$PLATFORM" \
  --provenance=false \
  --build-arg "CODEX_VERSION=$CODEX_VERSION" \
  --file "$ROOT/deploy/docker/session.Dockerfile" \
  --tag "$REGISTRY/$REPOSITORY:$TAG" \
  --load \
  "$ROOT"

docker push "$REGISTRY/$REPOSITORY:$TAG"

# The registry's own answer rather than the line `docker push` printed: this is what a
# kubelet resolves, and reading it back is also proof the manifest landed.
printf '%s/%s@%s\n' "$REGISTRY" "$REPOSITORY" \
  "$(aws ecr describe-images --repository-name "$REPOSITORY" \
       --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' \
       --output text)"
