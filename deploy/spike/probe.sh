#!/bin/sh
# Is the in-pod sandbox a security boundary on a hardened node, and what does it cost?
#
# The probe answers that by emitting one `key=value` finding per check, to stdout and to
# a file, and exiting non-zero the moment one of them fails. The pod's `restartPolicy:
# Never` plus that exit status is the spike's verdict; there is no separate reporter.
#
# Two shapes recur and both are deliberate:
#
#   A tool is proven present as its own finding before its result is allowed to mean
#   anything. A missing binary exits 127 and a kernel refusal exits non-zero too, so a
#   bare exit-status check reports "this kernel forbids user namespaces" for an image
#   that simply lacks util-linux -- the wrong verdict, and it fails toward "the
#   architecture does not work".
#
#   A refusal is never taken as evidence on its own. A write can be refused for many
#   reasons that have nothing to do with the rule meant to refuse it, so section 4c
#   asserts the rule is present in the argv `bwrap` was actually handed, separately from
#   asserting the write was refused. The argv is captured by shadowing `bwrap` on PATH
#   with a wrapper that records its arguments and then execs the real binary, which
#   reads what was compiled rather than what somebody intended to compile.
set -eu

OUT=/work/spike-report.txt
SHIM=/work/shim
ARGV=/work/argv.txt
WS=/work/ws
SOCK=/run/map/app-server.sock

: > "$OUT"
say() { printf '%s\n' "$*" | tee -a "$OUT"; }
die() { say "$*"; exit 1; }

mkdir -p "$SHIM" "$WS"

say "## 4a host requirements"
# I1 says the pod holds no credential. Kubernetes projects a ServiceAccount token by
# default, and on this cluster that token is also AssumeRoleWithWebIdentity material for
# the OIDC roles -- so a default this manifest does not override is an AWS credential
# sitting inside the artifact whose whole job is to prove the pod is a boundary. Asserted
# here rather than trusted from pod.yaml, because a manifest line is a claim and this is
# the measurement.
if [ -e /var/run/secrets/kubernetes.io/serviceaccount/token ]; then
  die "serviceaccount-token-absent=PRESENT"
else
  say "serviceaccount-token-absent=ok"
fi

# Seccomp filtering and unprivileged user namespaces interact, and which one is in force
# decides whether any sandbox can be built at all -- so the mode is recorded rather than
# assumed. 0 is no filter, 2 is a seccomp-bpf filter installed.
say "seccomp-mode=$(awk '/^Seccomp:/{print $2}' /proc/self/status)"
say "max-user-namespaces=$(cat /proc/sys/user/max_user_namespaces)"

command -v unshare >/dev/null || die "unshare-tool=MISSING-FROM-IMAGE"
say "unshare-tool=present"
if unshare --user --map-root-user true 2>/dev/null; then
  say "unshare-userns=ok"
else
  die "unshare-userns=REFUSED-BY-KERNEL rc=$?"
fi
# The mount namespace is the one bwrap actually needs; a userns it cannot mount into is
# no more use here than no userns at all, so it is asserted separately.
if unshare --user --map-root-user --mount true 2>/dev/null; then
  say "unshare-userns-mount=ok"
else
  die "unshare-userns-mount=REFUSED rc=$?"
fi

command -v bwrap >/dev/null || die "bwrap-tool=MISSING-FROM-IMAGE"
say "bwrap-tool=present"
say "bwrap-version=$(bwrap --version)"
if bwrap --unshare-user --unshare-net --ro-bind / / /bin/true 2>/dev/null; then
  say "bwrap-usable=ok"
else
  die "bwrap-usable=CANNOT-BUILD-SANDBOX rc=$?"
fi

say ""
say "## 4b the runtime's own limits, read off the binary"

say "codex-version=$(codex --version)"
# Which Linux backend the binary actually uses is not a setting to read but a fact to
# observe: the two historical toggles are both retired, so the answer is whatever the
# `bwrap` shim below does or does not catch.
say "feature-use-linux-sandbox-bwrap=$(codex features list | awk '/^use_linux_sandbox_bwrap/{print $2" "$3}')"
say "feature-use-legacy-landlock=$(codex features list | awk '/^use_legacy_landlock/{print $2" "$3}')"

cat > "$CODEX_HOME/config.toml" <<TOML
default_permissions = "map-spike"

# Evaluated most-specific-path-wins: the socket's deny overrides the write on the
# directory containing it, which overrides the read on the root.
[permissions.map-spike.filesystem]
"/" = "read"
"$WS" = "write"
"/run/map" = "write"
"$SOCK" = "deny"
# /work holds the argv capture -- the compiled policy that confines this very command --
# and the captured provider exchange, including the bearer header the recorder redacts on
# its way to the transcript but not on disk. Under "/" = "read" both were readable from
# inside the sandbox, which is a route around the redaction rather than a leak through it.
# $WS is more specific and keeps its write.
"/work" = "deny"
# Nothing should be here: the pod sets automountServiceAccountToken: false. Denied anyway,
# so the boundary does not depend on one line of a manifest a later slice will rewrite.
"/var/run/secrets" = "deny"

# Identical to map-spike but for one line: the socket is writable. It exists so the
# assertions below have a negative control -- a case whose expected answer is the
# opposite of the one beside it, under a policy that differs only in the rule being
# tested. Without it, swapping the bodies of an if/else in this script leaves every
# check in tests/spike/ green: they grade which words the probe can emit and which of
# them sit on a die, and swapping two branches changes neither.
#
# It inverts the /work deny rather than the socket's, because the socket is a regular file
# and a write grant on one makes the runtime panic -- measured as
# write-grant-on-a-regular-file below. So the control sits on a directory, which is the
# only shape a write grant can take.
[permissions.map-spike-inverted.filesystem]
"/" = "read"
"$WS" = "write"
"/run/map" = "write"
"$SOCK" = "deny"
"/work" = "write"
"/var/run/secrets" = "deny"

[mcp_servers.probe]
command = "/bin/true"
TOML

# The effective MCP timeouts are compiled-in and the CLI surfaces no way to read them:
# what it renders is the *configured* value, which is unset. That is the finding -- a
# platform that wants a known bound has to set one rather than inherit one.
say "mcp-startup-timeout-sec=$(codex mcp get probe --json | awk -F': ' '/startup_timeout_sec/{gsub(/[ ,]/,"",$2); print $2}')"
say "mcp-tool-timeout-sec=$(codex mcp get probe --json | awk -F': ' '/tool_timeout_sec/{gsub(/[ ,]/,"",$2); print $2}')"
# Written to a file and then searched, rather than piped into `grep -q`: grep closes the
# pipe on its first match and the CLI panics on the resulting EPIPE, which puts a Rust
# backtrace in the middle of the transcript this record is quoted from.
codex -c mcp_servers.probe.startup_timeout_sec=11 -c mcp_servers.probe.tool_timeout_sec=22 \
  mcp get probe --json > /work/mcp-override.json
if grep -q '"startup_timeout_sec": 11' /work/mcp-override.json; then
  say "mcp-timeout-override-accepted=ok"
else
  die "mcp-timeout-override-accepted=REFUSED"
fi

# The tool-output cap is not a setting. It is a per-model field of the runtime's own
# model catalog, so it arrives with the model rather than with the configuration, and
# there is no key to raise. Printed as the distinct values across the whole catalog with
# a count each, because one model disagreeing with the rest is the case that matters.
codex debug models > /work/models.json
say "tool-output-cap=$(python3 -c '
import collections, json, sys
models = json.load(open("/work/models.json"))["models"]
seen = collections.Counter(json.dumps(m.get("truncation_policy"), sort_keys=True) for m in models)
print("; ".join("%s x%d" % (policy, n) for policy, n in sorted(seen.items())))
')"
# Whether the cap is raisable decides whether ADR-019's mechanism exists, so it is
# measured rather than asserted. Each candidate key is set with `-c` and the catalogue
# re-read: if the policy moves, the key is the lever. Five plausible names is not proof
# that no key exists, so the finding names the count rather than claiming exhaustion.
CAP0=$(codex debug models | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["models"][0]["truncation_policy"], sort_keys=True))')
MOVED=""
TRIED=0
# Two shapes, because they fail differently. A scalar key would set the policy directly;
# `model_catalog_json` takes a path to a whole replacement catalogue. Testing only the
# first shape is what made an earlier reading of this probe conclude the cap was fixed.
codex debug models > /work/catalog.json
python3 - <<'PATCH'
import json
d = json.load(open("/work/catalog.json"))
for m in d["models"]:
    m["truncation_policy"] = {"mode": "bytes", "limit": 5000000}
json.dump(d, open("/work/catalog-raised.json", "w"))
PATCH
for KEY in tools.max_output_bytes truncation_policy model_truncation_policy \
           tools.truncation_policy shell.max_output_bytes; do
  TRIED=$((TRIED + 1))
  CAP1=$(codex -c "$KEY={\"limit\"=50000,\"mode\"=\"bytes\"}" debug models 2>/dev/null \
    | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["models"][0]["truncation_policy"], sort_keys=True))' 2>/dev/null || true)
  [ -n "$CAP1" ] && [ "$CAP1" != "$CAP0" ] && MOVED="$KEY"
done
for KEY in model_catalog_json model_catalog_path models_file; do
  TRIED=$((TRIED + 1))
  CAP1=$(codex -c "$KEY=\"/work/catalog-raised.json\"" debug models 2>/dev/null \
    | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["models"][0]["truncation_policy"], sort_keys=True))' 2>/dev/null || true)
  [ -n "$CAP1" ] && [ "$CAP1" != "$CAP0" ] && MOVED="$KEY -> $CAP1"
done
if [ -n "$MOVED" ]; then
  say "tool-output-cap-config-key=$MOVED"
else
  say "tool-output-cap-config-key=none-of-$TRIED-tried"
fi

say ""
say "## 4c the deny rule is in the compiled argv, and the write is refused"

REAL_BWRAP=$(command -v bwrap)
cat > "$SHIM/bwrap" <<EOF
#!/bin/sh
# Records the argv the runtime compiled, then runs the real thing unchanged.
# One argument per line. \$* joins on IFS, which makes an argument containing a space
# indistinguishable from two arguments and leaves the assertion below matching a
# substring of a flattened string rather than a rule with a shape.
printf '%s\n' "\$@" >> $ARGV
exec $REAL_BWRAP "\$@"
EOF
chmod 0755 "$SHIM/bwrap"

# Print the compiled deny rule as its three adjacent argv tokens -- the bind option, its
# source, and the masked path -- or print nothing. One reader for both assertions below,
# because they ask the same question and a substring match answered it wrongly in both:
# the socket path also appears inside a --setenv value and inside a bind of its parent
# directory, so `grep -q "$SOCK"` reports a rule that was never compiled.
deny_form() {
  awk -v sock="$SOCK" '
    $0 == sock && NR >= 3 && p2 ~ /^--[a-z-]*bind[a-z-]*$/ { print p2 " " p1 " " $0; exit }
    { p2 = p1; p1 = $0 }
  ' "$ARGV"
}
PATH="$SHIM:$PATH"
export PATH

test -e "$SOCK" || die "deny-target-missing=$SOCK"
test -s "$SOCK" && die "deny-target-not-empty=$SOCK"

: > "$ARGV"
if codex sandbox -P map-spike -- /bin/sh -c "printf allowed > $WS/ok.txt" 2>>"$OUT"; then
  say "write-inside-profile=ok"
else
  die "write-inside-profile=REFUSED rc=$?"
fi
# Emitted as two literal branches rather than one command substitution, because this
# value is asserted by name in tests/spike/test_report_shape.py and a source scanner
# cannot read a value it has to run a subshell to learn. The failure branch dies: if
# bwrap did not build the sandbox, the premise this whole spike exists to test is gone.
if [ -s "$ARGV" ]; then
  say "sandbox-backend=bwrap"
else
  die "sandbox-backend=not-bwrap"
fi

# This is the slice's load-bearing assertion. It was a substring match, which let it
# report ok while the property was false; deny_form() asks for the rule's shape instead.
DENY_FORM=$(deny_form)
if [ -n "$DENY_FORM" ]; then
  say "deny-rule-in-argv=ok"
  say "deny-rule-argv-form=$DENY_FORM"
else
  die "deny-rule-in-argv=ABSENT"
fi

# The deny on /work is only worth the line if it holds from inside. Read, not write: the
# risk these two paths carry is exfiltration -- argv.txt is the policy confining this very
# command, inbound.txt is the captured provider exchange with the bearer header in it. A
# `cat` that succeeds and returns bytes is the finding; the mask makes it return nothing.
if codex sandbox -P map-spike -- /bin/sh -c "test -s $ARGV" 2>/dev/null; then
  die "sandbox-cannot-read-own-policy=LEAKED"
else
  say "sandbox-cannot-read-own-policy=ok"
fi

if codex sandbox -P map-spike -- /bin/sh -c 'test -s /work/inbound.txt' 2>/dev/null; then
  die "sandbox-cannot-read-inbound-capture=LEAKED"
else
  say "sandbox-cannot-read-inbound-capture=ok"
fi

if codex sandbox -P map-spike -- /bin/sh -c "printf x >> $SOCK" 2>>"$OUT"; then
  die "write-outside-profile=ALLOWED"
else
  say "write-outside-profile=refused"
fi
# Nothing partly written: the file must still be the zero bytes the init container left.
if [ "$(wc -c < "$SOCK")" -eq 0 ]; then
  say "no-partial-write=ok"
else
  die "no-partial-write=FAIL bytes=$(wc -c < "$SOCK")"
fi

# The negative control. The same write, the same path, the same binary -- under a profile
# whose only difference is that the socket is writable. It must SUCCEED. Together with
# write-outside-profile above, which must fail, this pins the polarity of the enforcement
# rather than its vocabulary: inverting the branches of either check now contradicts the
# other, and no single edit can invert both consistently because the two run under
# different policies. A source scanner cannot establish this; only a second run can.
CTL=/work/polarity.txt
rm -f "$CTL"
# Under map-spike, /work is denied: this must fail.
if codex sandbox -P map-spike -- /bin/sh -c "printf x > $CTL" 2>>"$OUT"; then
  die "polarity-control=WRITABLE-UNDER-DENY"
# Under map-spike-inverted, the same path is granted write: the same command must succeed.
elif codex sandbox -P map-spike-inverted -- /bin/sh -c "printf x > $CTL" 2>>"$OUT"; then
  say "polarity-control=ok"
else
  die "polarity-control=DENIED-UNDER-PERMISSIVE-PROFILE"
fi
rm -f "$CTL"
if [ -e "$CTL" ]; then
  die "polarity-control-cleanup=FAIL"
else
  say "polarity-control-cleanup=ok"
fi

# A write grant on a regular file is not a narrower version of a write grant on a
# directory -- it is a crash. The runtime treats every write root as a project root and
# mkdirs .git and .codex inside it, so a grant naming a file panics before the confined
# command runs. MAP-10 compiles these profiles from tenant-shaped input; a rule naming a
# file takes the Turn down rather than over-granting, which is the safe direction but is
# still a Turn that cannot run. Measured with a throwaway profile so the finding is a
# fact rather than a guess about the one above.
if codex -c "permissions.probe-filegrant.filesystem={\"/\"=\"read\",\"$SOCK\"=\"write\"}" \
     sandbox -P probe-filegrant -- /bin/true 2>>"$OUT"; then
  say "write-grant-on-a-regular-file=accepted"
else
  say "write-grant-on-a-regular-file=refused-or-panicked"
fi

# A deny rule over a path that does not exist has been reported elsewhere to compile to
# nothing at all, which would make the enforcement fail open whenever the pod's ordering
# slipped. Measured here rather than believed, because the answer decides whether the
# ordering is load-bearing or merely tidy. The target is restored afterwards.
rm -f "$SOCK"
: > "$ARGV"
codex sandbox -P map-spike -- /bin/true 2>>"$OUT" || true
# Both failing branches `die`. They are the fail-open cases -- a deny rule silently
# dropped, and a write to the denied path succeeding -- and a probe that prints them and
# exits 0 hands back a Succeeded pod carrying the words that mean the boundary leaked.
# Whether anyone reads the transcript is then the only thing between the leak and the
# plan. The pod's exit status is the one signal nothing has to remember to check.
if [ -n "$(deny_form)" ]; then
  say "deny-rule-in-argv-target-absent=present"
else
  die "deny-rule-in-argv-target-absent=DROPPED"
fi
if codex sandbox -P map-spike -- /bin/sh -c "printf x > $SOCK" 2>>"$OUT"; then
  die "write-outside-profile-target-absent=ALLOWED"
else
  say "write-outside-profile-target-absent=refused"
fi
rm -f "$SOCK"
: > "$SOCK"

# The Egress Policy is a userspace proxy and not a network rule, so what the sandbox
# itself grants the process is worth knowing: a namespace with only a loopback device
# means nothing the sandbox spawns can reach a destination on its own.
say "sandbox-network=$(codex sandbox -P map-spike -- /bin/sh -c 'cut -d: -f1 /proc/net/dev | tail -n +3 | tr -d " " | tr "\n" ","')"

say ""
say "## 4e the provider-facing inbound exchange, read off the binary"

# A contract this platform receives and cannot renegotiate. Measured by pointing the
# runtime at a local recorder rather than at a provider: the recorder is the only way to
# see the request the runtime issues, and it can answer with a stream whose terminator
# and unknown events are chosen rather than hoped for. No credential is involved -- the
# key below is a literal the recorder ignores and never logs.
cat > /work/recorder.py <<'PY'
"""Stand in for a model provider and write down exactly what the runtime asked for.

Answers every request with the same three-event stream: a created event, an event whose
type no provider defines, and a completed event. What the runtime does with the middle
one is the finding -- there is no other way to see whether an unrecognized event is
ignored or fatal.

Authorization values are replaced with a placeholder before anything is written, so the
transcript can be committed. The scheme is kept because the scheme is the contract.
"""
import http.server
import json

CAPTURE = "/work/inbound.txt"
# An allowlist, and the inversion is the point. This file is read back into a transcript
# that gets committed, so a header whose value reaches it reaches the repository. A
# denylist of known secret-carrying names -- authorization, x-api-key, and so on -- is
# wrong by default in exactly the case that matters: the header nobody thought of. A
# provider adding `x-session-key`, a proxy adding `proxy-authorization`, or a future
# runtime version adding anything at all would each write a live credential into a
# committed file, and nothing here would notice.
#
# So the default is to withhold, and a value is emitted only if its name is on this list.
# Every header still appears in the capture by *name*, which is what keeps the findings
# below working -- the record can still say which headers the runtime sends, it just does
# not print values it was not told are safe. The cost of the inversion is that a genuinely
# interesting value shows up as <withheld> until someone adds it here, and that is the
# right direction to fail in.
PRINTABLE_HEADERS = {"accept", "content-type", "content-length", "host", "user-agent"}

# Authorization is the one header where the *scheme* is the finding and the credential is
# the rest of the line: the record needs to say the runtime authenticates with a bearer
# token, without saying which one.
SCHEME_ONLY_HEADERS = {"authorization", "proxy-authorization"}
STREAM = (
    'event: response.created\n'
    'data: {"type":"response.created","response":{"id":"resp_map","status":"in_progress"}}\n\n'
    'event: map.unknown.event\n'
    'data: {"type":"map.unknown.event","note":"no provider defines this"}\n\n'
    'event: response.completed\n'
    'data: {"type":"response.completed","response":{"id":"resp_map","status":"completed",'
    '"output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
).encode()


class Recorder(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        with open(CAPTURE, "a") as out:
            out.write("REQUEST-LINE %s %s %s\n" % (self.command, self.path, self.request_version))
            for name, value in sorted(self.headers.items()):
                lowered = name.lower()
                if lowered in SCHEME_ONLY_HEADERS:
                    value = value.split(" ")[0] + " <withheld>" if " " in value else "<withheld>"
                elif lowered not in PRINTABLE_HEADERS:
                    value = "<withheld>"
                out.write("HEADER %s: %s\n" % (name, value))
            out.write("BODY-BYTES %d\n" % len(body))
            try:
                out.write("BODY-KEYS %s\n" % json.dumps(sorted(json.loads(body).keys())))
            except ValueError as bad:
                out.write("BODY-UNPARSED %s\n" % bad)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(STREAM)))
        self.end_headers()
        self.wfile.write(STREAM)


http.server.HTTPServer(("127.0.0.1", 8099), Recorder).serve_forever()
PY

cat >> "$CODEX_HOME/config.toml" <<'TOML'

[model_providers.map-recorder]
name = "map recorder"
base_url = "http://127.0.0.1:8099/v1"
wire_api = "responses"
env_key = "MAP_RECORDER_KEY"
TOML

python3 /work/recorder.py &
RECORDER_PID=$!
sleep 2
: > /work/inbound.txt
if MAP_RECORDER_KEY=not-a-secret codex exec -c model_provider=map-recorder \
     --skip-git-repo-check 'hi' >/work/exec.txt 2>&1; then
  say "inbound-exchange-captured=ok"
else
  # die, not say: no inbound-* key is in the test's REQUIRED_KEYS, so a failed capture
  # would leave a Succeeded pod whose record is simply missing its whole 4e section --
  # the "a skipped check reads as a pass" failure this file was written against.
  die "inbound-exchange-captured=FAILED rc=$?"
fi
kill "$RECORDER_PID" 2>/dev/null || true

say "inbound-request-line=$(awk '/^REQUEST-LINE/{print $2" "$3; exit}' /work/inbound.txt)"
say "inbound-auth-header=$(awk -F': ' '/^HEADER authorization:/{print $2; exit}' /work/inbound.txt)"
say "inbound-accept=$(awk -F': ' '/^HEADER accept:/{print $2; exit}' /work/inbound.txt)"
say "inbound-body-keys=$(awk '/^BODY-KEYS/{sub(/^BODY-KEYS /,""); print; exit}' /work/inbound.txt)"
# The turn completed on a stream carrying an event type no provider defines, so the
# runtime ignores what it does not recognize rather than failing the turn.
if grep -q 'tokens used' /work/exec.txt; then
  say "inbound-stream-terminator=response.completed"
  say "inbound-unknown-event=ignored"
else
  say "inbound-stream-terminator=UNDETERMINED"
  say "inbound-unknown-event=UNDETERMINED"
fi

say ""
say "## 4d p -- pods per node, at the raised 5 MB output cap, not the default"

say "node-memory-total-kb=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)"
say "rss-kb-idle=$(awk '/VmRSS/{print $2}' /proc/self/status)"

# `p` is a memory-bound count, so the figure it rests on has to be a real high-water
# mark. Two are taken because each is wrong on its own:
#
#   cgroup memory.peak is the kernel's own high-water mark for everything in this
#   container since it started, exact and impossible to sample past. It over-reads for a
#   single command, because it also holds whatever the earlier sections cost.
#
#   The sampled sum of VmRSS across the tree isolates this one command but samples every
#   200 ms, so it under-reads a peak that lives for less than that. It is a floor.
#
# Together they bracket the answer, and the record states both rather than picking one
# and calling it `p`.
CGROUP_PEAK=/sys/fs/cgroup/memory.peak
if [ -r "$CGROUP_PEAK" ]; then
  say "cgroup-memory-peak-kb-before-5mb=$(($(cat "$CGROUP_PEAK") / 1024))"
else
  say "cgroup-memory-peak-kb-before-5mb=unavailable"
fi

codex sandbox -P map-spike -- /bin/sh -c 'head -c 5242880 /dev/urandom | base64' >/work/big.txt 2>/dev/null &
LOAD_PID=$!
PEAK=0
while kill -0 "$LOAD_PID" 2>/dev/null; do
  TOTAL=0
  for status in /proc/[0-9]*/status; do
    RSS=$(awk '/VmRSS/{print $2}' "$status" 2>/dev/null || true)
    [ -n "$RSS" ] && TOTAL=$((TOTAL + RSS))
  done
  [ "$TOTAL" -gt "$PEAK" ] && PEAK=$TOTAL
  sleep 0.2
done
wait "$LOAD_PID" || true
say "rss-kb-peak-5mb=$PEAK"
if [ -r "$CGROUP_PEAK" ]; then
  say "cgroup-memory-peak-kb=$(($(cat "$CGROUP_PEAK") / 1024))"
else
  say "cgroup-memory-peak-kb=unavailable"
fi
say "output-bytes-5mb=$(wc -c < /work/big.txt)"

say ""
say "probe=complete"
