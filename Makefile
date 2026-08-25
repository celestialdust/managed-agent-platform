# The commands this project is checked and shipped with, in one file.
#
# They are here rather than in a CI workflow because CI and a developer must run the
# same words. A workflow that spells its own `ruff check` is a second copy of a fact,
# free to drift from the one a person types -- and the drift is invisible until a
# change passes locally and fails in CI, or worse, the other way round. So
# `.github/workflows/ci.yml` calls these targets and defines no command of its own.
#
# Run `make` with no target for the list.

SHELL := /bin/sh
.DEFAULT_GOAL := help

# Every `aws`, `kubectl` and `terraform` call below inherits this. It is the single
# most common way a deploy goes wrong here: the commands succeed against whatever
# account the ambient credentials name, which may not be this platform's. Override on
# the command line for another account -- `make deploy AWS_PROFILE=map-prod`.
AWS_PROFILE ?= map-dev-agent

# Exported to the deploy and infrastructure targets, and NOWHERE ELSE. It was exported
# globally in the first version of this file, which put a profile name into `make test`'s
# environment -- and the offline suite constructs a boto session at composition time, so
# on any machine that does not have that profile the whole suite dies in
# `aioboto3.Session()` with ProfileNotFound. That is not hypothetical: the first run of
# this pipeline was `58 failed, 3807 passed, 112 errors`, and every one of them was that
# line. It passed locally because the profile exists locally, which is the exact failure
# a shared Makefile is supposed to prevent.
_NEEDS_AWS := preflight image image-session bootstrap deploy \
              deploy-control-plane deploy-tool-gateway deploy-model-gateway \
              infra-plan infra-apply test-live
$(_NEEDS_AWS): export AWS_PROFILE := $(AWS_PROFILE)

# `-rs` always, not only in CI. A green summary does not mean the tiers below it ran, and
# the skip reasons in this suite are written to say what was not checked; a flag that only
# CI passes is a reading only CI gets.
PYTEST := uv run --extra dev pytest -q -rs -p no:randomly

.PHONY: help setup lint format format-check types residue test test-live check gates \
        api-surface preflight image image-session bootstrap deploy \
        deploy-control-plane deploy-tool-gateway deploy-model-gateway \
        infra-plan infra-apply publish publish-dry clean

help: ## Show this list
	@echo "Managed Agent Platform"
	@echo
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-22s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  AWS_PROFILE=$(AWS_PROFILE)  (override on the command line)"

# ---------------------------------------------------------------- develop and check

setup: ## Install the toolchain and every dependency, from the lockfile
	uv sync --extra dev --locked

lint: ## Static analysis, including the layering rule
	uv run ruff check src tests

format: ## Rewrite source to the project's format
	uv run ruff format src tests

format-check: ## Fail if source is not formatted
	uv run ruff format --check src tests

types: ## mypy --strict over source, tests and migrations
	uv run --extra dev mypy --strict src tests migrations

residue: ## Fail on conflict markers a merge resolution should have removed
	uv run python tools/merge_residue.py

# `env -u` and not merely "this target does not export it": a developer with AWS_PROFILE
# set in their own shell would otherwise hit the CI failure locally, which is the same
# defect arriving by the other door. The offline suite has to give one answer on every
# machine, and an ambient AWS variable is what stops it.
test: ## The offline suite -- needs a Docker daemon, no cluster, no AWS
	env -u AWS_PROFILE $(PYTEST)

# Not part of `check`, and it must not become part of it: this tier places real pods in
# a real cluster and costs minutes and money. `make gates` lists what else it turns on.
test-live: ## The tier that talks to the live cluster (needs AWS credentials)
	MAP_CLUSTER_TESTS=1 $(PYTEST)

check: lint format-check types residue test ## Every gate, in the order CI runs them
	@echo "all gates passed"

gates: ## List every environment variable that can skip a tier of the suite
	@uv run python tools/skip_gates.py

api-surface: ## Print the REST surface the app actually serves
	@uv run --extra dev python tools/api_surface.py

# ---------------------------------------------------------------------- deploy

# Checked before anything is built or applied, because each of these fails later in a
# way that reads as a different problem: absent AWS credentials surface as a registry
# 401 mid-push, a kubectl context pointing elsewhere applies this platform to another
# cluster, and a stopped Docker daemon fails the build after the tag has been resolved.
preflight: ## Check credentials, cluster reachability and the Docker daemon
	@aws sts get-caller-identity --query Arn --output text \
	  || { echo "no AWS credentials for profile $(AWS_PROFILE)" >&2; exit 1; }
	@kubectl config current-context \
	  || { echo "kubectl has no current context" >&2; exit 1; }
	@kubectl get namespace map-dev -o name \
	  || { echo "namespace map-dev is absent; run 'make bootstrap'" >&2; exit 1; }
	@docker info --format '{{.ServerVersion}}' >/dev/null \
	  || { echo "the Docker daemon is not responding" >&2; exit 1; }
	@echo "preflight ok"

image: ## Build the platform image and push it to all three repositories
	sh deploy/docker/push-platform-image.sh

image-session: ## Build and push the Session pod image (needs CODEX_VERSION)
	sh deploy/docker/push-session-image.sh

bootstrap: ## Put the cluster objects a Session pod needs in place
	uv run python deploy/bootstrap.py

deploy-control-plane: ## Roll the control plane, running this commit's image
	uv run python deploy/platform.py control-plane

deploy-tool-gateway: ## Roll the Tool Gateway
	uv run python deploy/platform.py tool-gateway

deploy-model-gateway: ## Roll the Model Gateway
	uv run python deploy/platform.py model-gateway

# The order is not arbitrary and is not modelled anywhere else. The image has to be in
# the registry before a manifest can name its digest. `bootstrap` creates the namespace
# and the identities the workloads run as. The control plane carries the schema
# migration Job, and the Tool Gateway reads tables that Job creates -- so it goes
# second. The Model Gateway depends on neither and goes last.
deploy: image bootstrap deploy-control-plane deploy-tool-gateway deploy-model-gateway ## Everything, in order: image, cluster objects, three workloads
	@echo "the platform is serving; run 'make test-live' to exercise it"

# ------------------------------------------------------------------ infrastructure

# `plan` and never `apply` from a plain target. Terraform here owns the cluster, the
# nodegroup, RDS, the VPC and the buckets, and an apply that replaces one of those
# destroys running state. The apply target exists so the command is written down, and
# it refuses without CONFIRM so it cannot be reached by a typo or a tab-complete.
infra-plan: ## Show what Terraform would change -- read this before applying
	cd deploy/terraform && terraform plan

infra-apply: ## Apply the Terraform configuration (needs CONFIRM=yes)
	@[ "$(CONFIRM)" = yes ] || { \
	  echo "refusing: this can destroy or replace the cluster, RDS or the buckets." >&2; \
	  echo "read 'make infra-plan' first, then run: make infra-apply CONFIRM=yes" >&2; \
	  exit 1; }
	cd deploy/terraform && terraform apply

# ------------------------------------------------------------------------ publish

publish-dry: ## Show what a publish would contain, and what it would leave behind
	uv run python tools/publish.py --dry-run

publish: ## Push the shippable tree to the public remote (excludes the design record)
	uv run python tools/publish.py

clean: ## Remove local caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache
