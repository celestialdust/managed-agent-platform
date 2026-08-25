#!/bin/sh
# Build the platform image and put it in the three registries the manifests pull from.
#
# A node pulls from a registry and never from a developer's Docker daemon, so an image
# that exists only locally cannot start a Deployment. This is the command that closes
# that gap, and it is a command a person runs: CI checks this repository and does not
# deploy it, because the account it pushes to is shared and the credentials that reach
# it should stay with an operator rather than with anything that can trigger a
# workflow. `make image` is this script.
#
# One build, three pushes, one digest. A digest is content-addressed and carries no
# repository, so the three references printed at the end differ only in their
# repository and name the same bytes -- which is what makes "one image, three services"
# checkable rather than claimed. The three repositories exist and are IMMUTABLE
# (deploy/terraform/registry.tf); a fourth `map/platform` would be an AWS resource and
# therefore Terraform's under ADR-021.
#
# The tag names the one input the tree controls -- the commit -- and nothing else. There
# is no runtime-version component because this image takes no build argument: the
# platform services run no codex. It is not the image's identity either; the manifests
# carry digests, so the tag exists so that a person holding a digest can find out what
# built it.
#
# It refuses to run while a build input is uncommitted, because the tag would then name
# a commit that is not what was built. Only the inputs count: the context admits
# pyproject.toml, uv.lock, src/**, alembic.ini and migrations/**, and BuildKit reads the
# two files named below, so an uncommitted document elsewhere in the tree changes no
# byte of the image and is not grounds for refusing.
set -eu

PLATFORM=linux/amd64
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# The three repositories, in the order the services depend on each other. Written out
# rather than derived from a registry listing: the Session pod's repository and the
# spike repository are in that listing too, and a script that pushed to whatever it
# found would push this image over the Session pod's.
REPOSITORIES="map/control-plane map/tool-gateway map/model-gateway"

# Before the print-tag seam below, so the seam cannot be used to get a tag for a tree
# that would not be allowed to push one.
#
# The pathspecs are written out rather than held in a variable: an unquoted expansion is
# what would make one word into several, and shellcheck is right to refuse that.
uncommitted=$(git -C "$ROOT" status --porcelain -- \
  pyproject.toml uv.lock src alembic.ini migrations \
  deploy/docker/platform.Dockerfile \
  deploy/docker/platform.Dockerfile.dockerignore)
if [ -n "$uncommitted" ]; then
  echo "a build input is uncommitted, so the tag would name a commit that is not" >&2
  echo "what was built. Commit or stash these and run again:" >&2
  echo "$uncommitted" >&2
  exit 1
fi

TAG="git-$(git -C "$ROOT" rev-parse HEAD)"

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

# Nothing to do only when every repository already holds this tag. Checked before the
# build, and it still prints the digests, because the digests are the whole output
# anybody uses. A partial run -- the first repository pushed and the third not -- lands
# here on the next attempt and pushes only what is missing.
missing=
for repository in $REPOSITORIES; do
  present=$(aws ecr describe-images --repository-name "$repository" \
    --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' \
    --output text 2>/dev/null || true)
  if [ -z "$present" ] || [ "$present" = "None" ]; then
    missing="$missing $repository"
  fi
done

if [ -n "$missing" ]; then
  aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY"

  # --provenance=false, --load, and a separate `docker push` per repository. All three
  # parts are one decision: exactly one manifest is PUT under each tag. By default
  # buildx exports three objects for a one-platform build -- the image manifest, a
  # provenance attestation manifest, and an index over both -- and binds the tag to the
  # index; ECR then refuses one of the two children under that tag with a bare 400,
  # after the bytes have already reached the registry. Measured on this slice's own
  # probe builds, which both printed `exporting attestation manifest`, and recorded in
  # push-session-image.sh:88-107 from the run that put a 1430-byte orphan into the
  # Session pod's repository.
  #
  # Loaded into the local image store rather than exported straight from buildx, for
  # the same file's other finding: under Docker Desktop's `docker` builder driver that
  # export runs from buildkit AND from the daemon copy of the same build, which is two
  # PUTs of a tag that allows one.
  #
  # Built once, outside the loop. Three tags on one build is three names for one local
  # image, so the three pushes send one manifest and the digests come back equal.
  set -- --platform "$PLATFORM" --provenance=false \
         --file "$ROOT/deploy/docker/platform.Dockerfile"
  for repository in $REPOSITORIES; do
    set -- "$@" --tag "$REGISTRY/$repository:$TAG"
  done
  docker buildx build "$@" --load "$ROOT"

  for repository in $missing; do
    docker push "$REGISTRY/$repository:$TAG"
  done
fi

# The registry's own answer rather than the lines `docker push` printed: this is what a
# kubelet resolves, and reading it back is also proof each manifest landed.
for repository in $REPOSITORIES; do
  printf '%s/%s@%s\n' "$REGISTRY" "$repository" \
    "$(aws ecr describe-images --repository-name "$repository" \
         --image-ids "imageTag=$TAG" --query 'imageDetails[0].imageDigest' \
         --output text)"
done
