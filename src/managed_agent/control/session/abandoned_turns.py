"""Closing the Turns a dead control plane left open, so the Session works again.

A Turn is opened by `turn.submitted` and closed only by a `turn.completed` or a
`turn.failed` naming it (`core/session/turns.py::open_turn`). Both closing events are
written by this process -- the streaming loop in `session_shim/pod_channel.py` appends
whatever the pod sends, and `control/session/turn_execution.py::run_turn` appends the
failure when the dispatch raises. Neither runs inside the tenant's request any more, so
a hung-up client no longer costs a Turn its ending; what still runs neither is a control
plane killed mid-Turn, and `HttpPodDispatch.dispatch` says so in its own words: "This
does not cover a control plane that dies mid-Turn. No `finally` runs after that."

What that leaves is not a slow Turn, it is a **permanently wedged Session**. The open
Turn refuses the Session's next Turn, because `admit_turn` requires
`state.accepts_a_turn()` and the fold leaves the Session `RUNNING`. It refuses the
Session's archive, because `archive_session` answers `ArchiveRefused` while `open_turn`
names a Turn. And it pins the pod sweep next door, whose third guard leaves a Session
with an open Turn alone whatever the clock says. There is no call a tenant can make that
gets out of it. This sweep is the only thing that does.

**Four independent signals close a Turn, and any one is enough.** They cover the ways
the platform loses a Turn that it can actually observe, and they want very different
waits. Two of them are the originals:

  1. The Turn's pod was placed and is now gone -- `POD_GONE_GRACE_MS`, two minutes. This
     is the common crash: the pod was deleted with the node, or finished on its own and
     nobody collected it. Two minutes is short because the evidence is strong; what
     makes it strong is the precondition below rather than the wait.
  2. The Turn's own report says its runtime went quiet -- `STUCK_IDLE_MS`, ten minutes.
     This is the wedge the pod signal cannot see: the pod is present and `Ready` and the
     process inside has stopped speaking to the shim. The pod is the only thing that can
     see that, so the pod is what says so.

**There was a third and it has been removed: a Turn is no longer closed for its age.**
Until 2026-08-26 any Turn open longer than sixty minutes was closed whatever its pod was
doing. That is the rule a clock can express, and it is the wrong rule, because an agent
run has no natural length. A real delegating review was observed thirty-eight minutes
into one Turn with forty-three live reviewer threads and `idle_ms` in the hundreds, and
the ceiling twenty-two minutes away; it finished at forty-four minutes and lost nothing.
What made that near miss worth acting on rather than shrugging at is that the phases it
had not yet reached -- adjudication, extraction, appraisal, report -- are the longest in
the run, and nothing ships until the Turn boundary, so the hour would have taken all of
it. Age was never evidence of death; it was a proxy for one, from before the platform
had the real signal. Signal 2 is that signal, and it reads what the pod says instead of
guessing from a clock.

**What that gives up, stated plainly.** The old ceiling was also the backstop for two
failures nothing else here covered, and a third has since been measured. A Turn stuck in
placement has no pod and no `turn.started`, so signal 1 has nothing to rest on. A shim
that dies with its pod still `Ready` simply stops reporting, and stopping is exactly
what signal 2 refuses to read, for the reason below. And a runtime that *retries* a
provider it cannot reach resets signal 2's clock with every attempt, which was measured
live and is written out below. Each wants its own bound, and a bound on placement, a
bound on an agent's thinking and a bound on a retry loop are not the same number --
which is the whole reason one constant should never have served even two of them.

**The first of those three is now signal 3**, below: a Turn seen with no pod and no
`turn.started` across `PLACEMENT_DEADLINE_MS` of sweeps is closed as
`RUNTIME_DID_NOT_START`. It gets its own constant and its own count for the reason the
paragraph above gives, and its number is sized against the 660-second cold placement
rather than against anything the other two signals use.

**The second is now signal 4**, also below: a live pod whose progress reports stopped
advancing across `REPORTS_CEASED_MS` is closed as `RUNTIME_LOST`. What makes it safe is
stated with the constant -- it reads a report sequence that *moved and then stopped*,
never an absence -- and what makes it possible at all is that the pod appends its own
reports on its own timer, so their absence still means something once the control plane
that used to read them is gone. **The remaining one -- a runtime looping on retries --
is still uncovered.**

**Signal 3 is a clock, and it is the one place here that resembles the ceiling.** The
difference that makes it defensible is what the clock is measuring. The ceiling bounded
a Turn's whole life, so it stood between a working agent and its own work, and the
longer and more valuable the run the more likely it was to fire -- which is how it came
within twenty-two minutes of killing a forty-four-minute review. This clock bounds only
the interval before a pod exists, during which the agent has by definition done nothing
and can lose nothing. No amount of legitimate work can push a Turn past it, because no
work has started. That is the property the ceiling never had, and it is why a deadline
is the right instrument here and was the wrong one there.

**Signal 2 acts on a report and never on the lack of one**, which is a load-bearing
asymmetry rather than a nicety. `runtime_image` is tenant-supplied and digest-pinned,
and nothing ever migrates a tenant off an old digest, so pods that emit no progress at
all are a permanent population rather than a transitional one. Reading their silence as
a signal would close every Turn they run, and would present to that tenant as the
platform killing working agents. That is why the asymmetry is kept even now that it
leaves silence uncovered: until 2026-08-26 the ceiling caught a silent Turn an hour
late, and today nothing catches it at all. The fix is a signal that can tell reports
*ceasing* from reports *never starting* -- a Turn that reported and then stopped is
distinguishable from one that never reported, and only the first is evidence of a
death. That is signal 4, and it is built. Reading plain silence would not have been
that fix; it would be the tenant-killing rule wearing its clothes, which is why signal 4
keys its count on the report *sequence* and leaves a Turn that never reported alone for
ever.

**`idle_ms` is a silence clock, any frame at all resets it, and there is a real failure
that survives it.** Measured 2026-08-26 against a live pod cut off from the model
gateway at the network layer -- the pod stayed `Running` and `Ready`, its shim kept
answering, and its runtime blocked on a model call that could never be answered:

    frames=37  idle=131118  answer_bytes=114     <- silent, climbing
    frames=38  idle=  5836  answer_bytes=114     <- one frame, clock reset
    frames=38  idle=155835  answer_bytes=114     <- silent, climbing again
    frames=39  idle= 25835  answer_bytes=114     <- one frame, clock reset

The runtime retried roughly every 170 seconds and each attempt produced exactly one
frame carrying no answer at all. Watched for fifteen minutes: `answer_bytes` never left
114 across thirty-one consecutive reports, `frames` reached 42, and `idle_ms` reset six
times with a maximum over the whole run of **165 359** -- less than a third of
`STUCK_IDLE_MS`, and not trending toward it. So **this Turn is never closed by this
signal**, and since the ceiling was removed it is never closed by anything. The Session
is wedged for good; the observation was stopped, the Turn was not.

That is not this signal misbehaving. It measures silence and the runtime was not
silent; it is the precise shape of what silence cannot see. Stated here rather than
left to be rediscovered, because it is the one case where removing the ceiling made
coverage strictly worse: an hour was a bad bound on a working agent, and it did close
this.

`answer_bytes` is the field that told the truth throughout, and reading it is still not
obviously right: a Turn legitimately running tools also produces frames and no answer
bytes for minutes at a time. What separates the two here is *rate* -- a healthy
delegating run produces frames continuously, this produced one per 170 seconds -- and
one measurement of each is not enough to choose a threshold that will not kill working
agents. Naming the gap is what today's evidence supports. Closing it wants either more
measurements or a distinction the shim cannot currently draw, and guessing at it would
repeat exactly the mistake the ceiling was.

**And the distinction may not belong here at all.** Every Session's model call goes
through the model gateway, which sees the upstream's own behaviour directly -- a
provider that hangs or refuses is a fact that component holds, not one this sweep has
to infer from frame timings. A real provider stall is therefore visible one layer down
with no threshold to guess at. The wedge measured above is the one case it would *not*
see, because the egress was cut before the gateway: that is a cluster fault affecting
every Session on that path rather than one Turn misbehaving, and it wants an operator
rather than a sweep. Worth saying because it means the missing bound is probably a
model-gateway decision that this file then reads, and building a frame-rate heuristic
here would be the wrong layer as well as the wrong evidence.

**The precondition that makes signal 1 safe is `turn.started`, and it is doing two jobs
at once.** A Turn that has not yet been given a pod is a Turn with no pod, and telling
that apart from a Turn whose pod died is the whole difficulty here: a first Turn's
placement was measured holding for 660 seconds while an autoscaled node arrived, and
during every second of it the Turn is open and no pod exists. A two-minute grace does
not save that Turn -- 120 is less than 660 -- so the grace is not what protects it. The
event is. `turn.started` reaches this log only by being streamed out of the pod and
appended by `pod_channel.py`'s line loop, so its presence proves a pod was created,
dialled, and answered. A Turn still in placement has `session.placing` and nothing else,
and can never reach signal 1 at all.

The same event is what lets this ask the cluster by Session and not by pod identity. A
Session pod's name is a pure function of its Session, so every Turn of a Session
produces a pod at the same name and a listing keyed on that name alone can be reading
the *previous* Turn's pod during its deletion grace. That ambiguity cannot survive
`turn.started`: a Session that is `RUNNING` refuses a second Turn
(`SessionState.accepts_a_turn` is `self is SessionState.IDLE`), so no second placement
runs while this Turn is open; and a placement creates the next pod only after waiting
for the old object to be **absent**, because "absent is the only state in which the API
server will accept a create at this name" (`KubernetesPodRunner._make_way_for_the_next_
pod`). So once this Turn's pod has answered, the at-most-one pod object the cluster can
hold at that name is this Turn's pod, and its absence is this Turn's pod being gone.

**Idempotent under two replicas, by construction and not by a lock**, which is the same
property the pod sweep next door has and for a related reason. Every input is state both
replicas read; `open_turn` matches terminal events by `turn_id`, so a second
`turn.failed` for a Turn already closed pops nothing and changes no fold; and the append
is preceded by a fold that stands down on a Turn nothing has left open. Two replicas
racing therefore cost at most one redundant ending event on a Turn that was ending
anyway -- the same bounded redundancy `lifecycle._end_and_release` documents and accepts
platform-wide.

What that fold cannot do is make the race impossible, and nothing here pretends it can:
both replicas may fold before either appends. It is the *bound* that makes this
acceptable rather than an exclusion -- one excess event per racing sweep on a Turn that
was ending anyway, never one per tick, because every later sweep folds and stands down.

**This sweep does not reclaim the pod, and that is a boundary not an oversight.**
Reclaiming pods is `reaper.py`'s single job, and it was already willing to reclaim this
one -- its third guard was the only thing stopping it. Closing the Turn is what lifts
that guard, so the pod is collected by the sweep that owns pods, on its next pass, from
its own guards. Two sweeps deleting pods would be two answers to one question.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, assert_never
from uuid import UUID

from managed_agent.control.session.lifecycle import close_abandoned_turn
from managed_agent.control.session.placement import PlacedPods, PodPhase
from managed_agent.core.ids import Seq, SessionId, TurnId
from managed_agent.core.ports import Clock, EventLogAppend, EventLogRange, EventRecord
from managed_agent.core.session.turns import open_turn
from managed_agent.core.vocabulary import turn

_LOG = logging.getLogger(__name__)

POD_GONE_GRACE_MS: Final = 2 * 60 * 1000
"""How long this Turn's pod must have been gone before the Turn is closed: two minutes.

Short, and it can afford to be, because the wait is not what carries the safety here.
`turn.started` is: a Turn that has not been placed yet cannot reach this branch at all,
so the wait is never standing between a Turn and its own 660-second cold start. What
remains for it to cover is a single unlucky read of the cluster, and two minutes of
consecutive sweeps is a generous margin against that.

**Counted from the first sweep that saw the pod missing, held in memory, and the memory
can only ever delay a close.** A replica that restarts forgets, and then waits two
minutes again from its next sweep; two replicas each keep their own count and each
arrive at the same close. Nothing about losing the count can make a Turn close *sooner*
than two observed minutes of absence, which is the direction that matters -- the failure
of this store is a Turn that stays open a little longer, never a live Turn killed.

Not read from the environment, for the reason `IDLE_GRACE_MS` gives next door: a number
that could be chosen per deployment is a number nobody had to decide, and this one has a
cost on both sides that somebody has to have accepted.
"""

PLACEMENT_DEADLINE_MS: Final = 22 * 60 * 1000
"""How long a Turn may sit with no pod and no `turn.started` before it is closed.

Twenty-two minutes, and the number is doing one job: clearing the worst placement this
platform has ever measured by a margin wide enough that the measurement not being a
proven maximum does not matter. A cold placement was measured holding **660 seconds**
while an autoscaled node arrived, and through every second of it the Turn is open with
no pod -- which is byte-for-byte the state this bound exists to close. Twice that
measurement is the margin, and `tests/control/test_an_abandoned_turn_is_closed.py::
test_the_placement_deadline_clears_the_worst_placement_ever_measured` refuses a value
that does not hold it.

**Long on purpose, because the two errors are not symmetric.** Too long leaves a Session
idle a few extra minutes and its tenant able to do nothing about it in the meantime. Too
short kills a Turn that was about to run, reports `runtime_did_not_start` for a runtime
that was seconds from starting, and does it precisely to the tenants waiting on scarce
capacity -- the ones whose placements are slowest and who can least afford the loss.
That is the failure the removed age ceiling was retired for, and picking this number
tight enough to reintroduce it would be repeating the mistake in a smaller font.

**What makes it safe is not the margin, it is that no live placement can still be
running.** `KubernetesPodRunner` bounds its own wait: `_SCHEDULING_TIMEOUT_SECONDS` is
420 for a node to arrive and `_READY_TIMEOUT_SECONDS` is 300 for the pod to come up
afterwards, and the second is *reset* when the pod is first seen scheduled rather than
added to a shared budget -- so 720 seconds is the longest a placement can hold before it
gives up and the request path fails the Turn itself. That is ten minutes short of this
deadline. A Turn still open and still unplaced when this fires therefore has no process
waiting on it: whatever was placing it died without appending anything, which is the
wedge this whole module exists for. `tests/control/test_an_abandoned_turn_is_closed.py::
test_the_placement_deadline_outlasts_the_longest_a_placement_can_run` refuses a value
that stops clearing those two.

Without that argument the bound would be closing Turns out from under live placements,
and the tenant would get `runtime_did_not_start` for a pod that then started, answered,
and streamed a completed Turn into a log that already called it failed.

The pod object narrows it further. A placement creates the object and *then* waits, so a
pod that is merely unscheduled or not yet ready is still listed and still clears this
count. The absence this deadline measures is therefore not "placement is slow" but "no
object exists at this Session's name at all", which during a healthy placement is a
window of seconds.

**Counted from the first sweep that saw the Turn unplaced, in memory, and losing the
count can only ever delay a close** -- the same property `POD_GONE_GRACE_MS` records
next door, and it holds here for the same reason: a restarted replica starts its count
again from its next sweep, and two replicas each reach the same close independently.
Nothing about losing that memory closes a Turn sooner than the deadline's worth of
observed absence.

Deliberately *not* the same constant as `POD_GONE_GRACE_MS`, which is two minutes. That
one can afford to be short because `turn.started` carries its safety, so its wait never
stands between a Turn and a cold start. This one has no such event to lean on -- the
absence of `turn.started` is the very condition it fires on -- so the clock is the only
protection there is, and it has to be sized for the worst wait rather than for an
unlucky read.
"""

STUCK_IDLE_MS: Final = 10 * 60 * 1000
"""How long a Turn's own report must say the runtime has been quiet: ten minutes.

**Read from a report, never from its absence, and that distinction is the whole design
of this signal rather than a caveat on it.** `runtime_image` is tenant-supplied and
digest-pinned (`core/registration/environment.py`), and nothing in the platform ever
migrates a tenant off an old digest. A pod built before the emitter reports nothing at
all, for ever, and there will be such pods as long as any tenant wants one. So silence
here is not a transitional state that drains after a rollout -- it is a permanent and
entirely healthy population, and a sweep that read silence as a signal would close every
Turn belonging to it.

**Ten minutes, and the evidence behind it is a real workload rather than a synthetic
one.** The first number this rested on came from a pod told to run `sleep 150`, which
reported `idle_ms` between 16 and 24 seconds throughout. That is the *quietest possible*
Turn -- no tool calls, no long model reasoning, a shell wait and nothing else -- so it
sets a floor on the healthy range and says nothing about the top of it. Against an
actual agent workload (a literature-review pipeline: search, dedup, screening) the
observed maximum is **72 053 ms**, and that gap had `frames` frozen across three
consecutive reports either side of it -- a genuine seventy-second pause in the runtime,
mid-work, on a Turn that recovered on its own and finished normally. So ordinary healthy
work reaches roughly one eighth of this threshold, not one twenty-fifth.

Ten minutes still clears that by about eight times, which is why the number stands.
Eight is a defensible margin and no healthy window anywhere near 600 seconds has ever
been seen. But **every time the workload got more real this number went up** -- 24s
synthetic, then 31s, then 42s, then 72s, each from a more realistic run than the last
-- so treat it as a lower bound on the healthy maximum rather than as the maximum
itself. **If a measurement ever shows a healthy gap within about half of this, the
threshold is too close and wants raising -- not the measurement discarded.**

One thing this measurement kills, stated because it is the tempting wrong lesson. The
retired inter-byte deadline was 120 seconds, and the reason it was wrong is that it
watched the stream the pod *published* rather than the one the runtime *speaks on* --
not that healthy work never went quiet for long. On the corrected stream healthy work
has now been seen quiet for 72 seconds, which is 60% of that deadline, with the longest
reasoning phases still ahead of it. **A deadline of 120 seconds would not be safe on
this stream either**, and anyone tempted to resurrect one on the corrected signal
should start from this paragraph rather than from the retired number.

It is not derived from any transport or wall-clock number and must not become so: a
reporting threshold and a give-up threshold answer different questions. Tying them is
exactly the mistake that produced the removed ceiling -- `httpx` needs a finite read
timeout, that number became the dispatcher's give-up point, and the give-up point then
became the maximum length of an agent run. Three unrelated questions, one constant.

**What this signal can and cannot see, stated because the temptation is to over-read
it.** `idle_ms` is measured from the last frame the shim *received* from the runtime, so
this catches a runtime that stopped talking while its pod stayed up -- the wedge no
pod-shaped signal can see, since the pod is present and answering. It does not
catch a runtime that chatters while making no progress: frames arrive whether or not
work is happening, so liveness and progress are different facts and only the first is
measured here. And if the shim itself dies the reports simply stop, which this
deliberately treats as nothing at all -- reading an absence here is the one thing that
would close every pre-emitter pod on the platform. That case belongs to
`REPORTS_CEASED_MS`, which reads reports *ceasing after having started*: distinguishable
from never having reported, and therefore safe to act on where plain silence is not.
"""

RUNTIME_SILENCE_DEADLINE_MS: Final = 60 * 60 * 1000
"""How long the control plane will hold a Turn's socket open with nothing arriving.

**This is not a bound on how long a Turn may run, and it used to be.** Until
2026-08-26 the same number was `TURN_CEILING_MS`: the sweep closed any Turn older than
it, and the dispatcher gave up on one at the same moment, whatever the pod was doing.
That came within twenty-two minutes of killing a real delegating Turn -- forty-three
live reviewer threads, seventy-four thousand answer bytes, `idle_ms` in the hundreds --
whose remaining phases were its longest, and nothing ships until the Turn boundary, so
the hour would have taken the lot. An agent run has no natural length, so no wall-clock
age can distinguish a long one from a dead one. That
job now belongs to `STUCK_IDLE_MS`, which reads what the pod says about itself instead
of guessing from a clock.

What is left is narrower and defensible: `httpx` needs a finite `read` timeout or a
half-open socket holds a connection for the life of the process. An hour is chosen to
be far above any silence a working pod produces -- the longest measured is about seven
minutes on this stream -- so it bounds a leak and never a Turn.

**It is deliberately no longer shared with anything that ends a Turn.** The old comment
here argued that one constant serving both the sweep and the dispatcher was the whole
point, so the two could not drift. That was true and it was the wrong thing to make
true: it meant the transport's need for a finite timeout silently set the maximum
length of an agent run.
"""

REPORTS_CEASED_MS: Final = 20 * 60 * 1000
"""How long a live pod may append no progress report before its Turn is closed.

**The third signal, and the one that catches a dead shim.** The other two cannot: the
cluster says the pod is there, and `idle_ms` is a number a *report* carries, so a shim
that stops reporting freezes it at whatever the last report said rather than letting it
grow. Between them a Turn whose shim died under a pod that stayed Running was invisible
for ever, and this module's docstring carried it as a named hole until this was built.

What makes the absence readable at all is that the pod appends its reports **itself**,
on its own timer, with no control plane in the path. So after a control plane dies --
the situation this whole module exists for -- a healthy pod's reports keep landing and
a dead shim's stop, which is the one stream that still distinguishes them.

Sized against the emitter and not against a clock. `_PROGRESS_INTERVAL_S` is thirty
seconds and a failed append is dropped rather than retried, deliberately, so a database
blip eats a run of reports and the next one carries everything the lost ones did. Twenty
minutes is forty of those intervals: a run of dropped reports long enough to reach it is
an outage, and during a log outage this sweep cannot read the log either and returns
`THE_SWEEP_COULD_NOT_DECIDE` rather than closing anything.

**Longer than `STUCK_IDLE_MS` on purpose, and the ordering is the argument.** That one
reads the pod's own measurement of its runtime; this one infers from an absence, and the
count starts at the first sweep that *observed* the silence rather than when it began --
the log port hands back no timestamp, so the measured silence always understates the
true one. A weaker signal measured from a later start is given the longer rope.
"""

_OPEN_TURN_LOOKBACK_MS: Final = 24 * 60 * 60 * 1000
"""How far back the cross-Session scan looks for Turns that were never closed.

The Event Log port hands back no timestamp on an event -- `EventRecord` carries a
sequence, a type and a payload and nothing else -- so "how old is this Turn" is not a
question that can be asked of one Session's log. What can be asked is which Sessions
appear in a window, which is the same shape the pod sweep's recency scan uses, and this
window is how a Turn old enough to judge is found at all.

A day rather than the ceiling itself, so a Turn crossing the ceiling has twenty-three
hours of sweeps in which to be noticed rather than the width of one tick. The scan is
narrow enough to afford it: it asks for three turn-boundary types and gets one row per
Turn submitted, started or ended -- not the stream, which is where the volume is.

**A Turn open for longer than this window is not found by this scan, and that residue is
real.** It needs the control plane to have swept nothing for the better part of a day,
which is the case a startup reconciliation would cover. What partly covers it already is
that a Session still holding a pod is a candidate whatever its log's age, because the
cluster listing supplies it -- so the uncovered case is narrower still: a Turn open more
than a day whose pod is also gone.
"""

_TURN_BOUNDARIES: Final = (
    turn.TURN_SUBMITTED,
    turn.TURN_COMPLETED,
    turn.TURN_FAILED,
)

_SWEEP_READS: Final = (
    turn.TURN_SUBMITTED,
    turn.TURN_STARTED,
    turn.TURN_COMPLETED,
    turn.TURN_FAILED,
)
"""Every type the per-Session part of a sweep folds, and nothing else.

The prescreen's three plus `turn.started`, which the prescreen must not have -- its one
rule is that a Session whose latest boundary row is a submission has a Turn open, and a
start would sit above the submission and break it. Here there is no latest-row rule to
break: `open_turn` acts on submissions and terminals and ignores everything else, so a
start passes through it untouched and is read instead by `started`, which is the only
thing that tells a pod that never came from one that came and went.

Naming the set is what keeps the read honest. The sweep used to page the Session's whole
log and pick these out in Python, which cost a row for every token-level delta the Turn
had emitted -- and which quietly made every new event type the sweep's problem. What it
folds is now the same list the store is asked for.
"""
"""The three types that open and close a Turn, and the whole basis of the prescreen.

`turn.started` is deliberately absent even though this module reads it elsewhere: it
neither opens nor closes a Turn, so a Session whose latest boundary row is a start would
be indistinguishable from one whose latest row is a submission, and the prescreen's one
rule -- latest boundary is a submission means a Turn is open -- would stop being true.

The stream types are absent for the reason the pod sweep gives for leaving them out of
its own scan: including them would return every token of every Turn in the window to
answer a question about which Sessions exist.
"""


_UNKNOWN_TURN: Final = TurnId(UUID(int=0))
"""The Turn named by an outcome the sweep could not decide.

An outcome is returned even for a Session whose log would not read, because a Turn this
sweep could not judge is still wedging its Session -- but the identifier is exactly what
could not be read, so it is a nil uuid rather than a guess. `TurnOutcome.turn_id` is not
optional for that one case, which would make every caller of the common path unwrap it.
"""


def _sessions_in(rows: Sequence[object]) -> frozenset[SessionId]:
    """The Sessions appearing in a window's rows, refusing a row without one.

    Refused rather than skipped, for the reason the neighbouring sweep gives: a row
    shape this cannot read would silently empty the set, and an empty set here reads as
    "no Session submitted recently" -- which is the input that makes the ceiling fire on
    every open Turn at once.
    """
    out: list[SessionId] = []
    for row in rows:
        session_id = getattr(row, "session_id", None)
        if session_id is None:
            raise TypeError(
                f"a turn-boundary row carries no session_id: {type(row).__name__}"
            )
        out.append(SessionId(session_id))
    return frozenset(out)


class TurnVerdict(StrEnum):
    """What this sweep decided about one open Turn. Every member is reached.

    A closed set rather than a log sentence, because a caller counts them: "four kept
    because their pods are still there" and "four kept because the sweep could not
    decide" describe a healthy platform and a broken one, and a sentence makes them the
    same line.

    Four of the eight keep the Turn open. The sweep closes only where it has a positive
    reason, and every reason it declines is named, so a Turn somebody expected to be
    closed can be explained without reading a cluster.

    There is deliberately no "the Turn is still young" member, and since the ceiling was
    removed there is no member for age at all: age neither keeps a Turn here nor closes
    one. Every Turn the sweep leaves alone is left alone because of what its pod is
    doing, and every Turn it closes is closed for the same kind of reason. A member
    standing for youth would be a second name for whichever of the three pod cases
    applied, and the pod case is the one that says why.
    """

    THE_POD_WAS_NEVER_PLACED = "the_pod_was_never_placed"
    THE_POD_IS_STILL_THERE = "the_pod_is_still_there"
    THE_POD_HAS_ONLY_JUST_GONE = "the_pod_has_only_just_gone"
    THE_SWEEP_COULD_NOT_DECIDE = "the_sweep_could_not_decide"

    ITS_POD_IS_GONE = "its_pod_is_gone"
    ITS_RUNTIME_STOPPED_TALKING = "its_runtime_stopped_talking"
    IT_NEVER_GOT_A_POD = "it_never_got_a_pod"
    ITS_REPORTS_STOPPED_ARRIVING = "its_reports_stopped_arriving"

    def closed_the_turn(self) -> bool:
        """Whether this verdict ended the Turn.

        The verdict answers this rather than each caller re-deriving it from a set of
        members, so a member added later cannot read as a keep to one reader and a close
        to another. Written as a match with an `assert_never` tail and no default arm,
        which is what makes a new member fail `mypy --strict` until it has chosen a side
        -- falling through to a default would silently read as a keep.
        """
        match self:
            case (
                TurnVerdict.ITS_POD_IS_GONE
                | TurnVerdict.ITS_RUNTIME_STOPPED_TALKING
                | TurnVerdict.IT_NEVER_GOT_A_POD
                | TurnVerdict.ITS_REPORTS_STOPPED_ARRIVING
            ):
                return True
            case (
                TurnVerdict.THE_POD_WAS_NEVER_PLACED
                | TurnVerdict.THE_POD_IS_STILL_THERE
                | TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE
                | TurnVerdict.THE_SWEEP_COULD_NOT_DECIDE
            ):
                return False
            case _ as unreachable:
                assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """One open Turn, and what the sweep did about it."""

    session_id: SessionId
    turn_id: TurnId
    verdict: TurnVerdict


class TurnBoundaryScan(Protocol):
    """The one cross-Session question this sweep asks of the Event Log.

    Narrower than the store that answers it, and declared here rather than shared with
    the pod sweep's `RecentActivity` over the same method: what the two have in common
    is one adapter's capability, not a rule either could change, and this one reads two
    fields off a row where that one reads a single field. A double for this sweep
    implements one method instead of a whole log.
    """

    async def lifecycle_events_between(
        self, types: Sequence[str], from_ms: int, to_ms: int
    ) -> Sequence[object]:
        """Events of these types appended after `from_ms` and at or before `to_ms`.

        The rows come back carrying a `session_id`, a `seq` and a `type`, which are the
        three fields read here. Typed as `object` for the reason the neighbouring scan
        gives -- naming a row type would make every double reproduce fields this never
        touches -- and the fields are read off the row by
        `_latest_boundary_per_session`, which refuses a row missing any of them rather
        than guessing.
        """
        ...

    async def turn_boundaries_of(
        self, session_id: SessionId, types: Collection[str]
    ) -> Sequence[EventRecord]:
        """This Session's events of these types, in sequence order.

        Typed as records rather than as `object`, unlike the scan above, because these
        rows are folded by `open_turn` and `started` and both read the payload. A double
        that hands back its own events already satisfies this.

        Unbounded in row count on purpose: it is bounded by how many Turns the Session
        has run, and a cap would risk truncating away a submission -- leaving
        `open_turn` to answer "nothing is open" for a Session that is wedged, the one
        wrong answer that makes a sweep silently useless.
        """
        ...

    async def latest_progress_of(
        self, session_id: SessionId, type_: str, turn_id: str
    ) -> Sequence[EventRecord]:
        """This Turn's newest event of this type, as a sequence of nought or one.

        A sequence and not a record-or-None so that `latest_idle_ms` and
        `latest_report_seq` keep the signature they already have: handed one row they
        read it, handed none they answer None, which is what "this Turn never reported"
        has always meant to them.
        """
        ...


def started(events: Sequence[EventRecord], turn_id: TurnId) -> bool:
    """Whether this Turn's pod ever answered.

    True exactly when a `turn.started` naming this Turn is in the log. That event is not
    written by the control plane on its own account: it is translated from the Agent
    Runtime's own `turn/started` notification and appended by the line loop in
    `session_shim/pod_channel.py`, which is reading the pod's response body. So it is in
    the log only if a pod was created for this Turn, was dialled, and replied.

    That is what tells "the pod was never placed" apart from "the pod was placed and is
    now gone", and the two are otherwise identical from outside: both are an open Turn
    with no pod in the cluster. Matched on `turn_id` rather than on the type alone,
    because a Session's log holds the started events of every Turn it has ever run, and
    the question is about this one.
    """
    return any(
        event.type == turn.TURN_STARTED and event.payload.get("turn_id") == str(turn_id)
        for event in events
    )


def latest_idle_ms(events: Sequence[EventRecord], turn_id: TurnId) -> int | None:
    """What this Turn's most recent progress report said, or None if it made none.

    **None and zero mean opposite things and the type is what keeps them apart.** None
    is "this Turn has never reported", which is the permanent condition of every pod
    running an image built before the emitter, and is never evidence of anything. Zero
    is "the runtime spoke just now", the healthiest reading there is. A function
    returning `0` for both would make a silent pod indistinguishable from a maximally
    live one, and it would read as the second -- the wrong direction to be wrong in,
    because it hides a pod nobody can see rather than closing one that is fine.

    Scanned backwards, and the first match wins. The reports are a running commentary
    and only the last one describes the present: a Turn that idled through one long
    model call and then resumed work must not be condemned by the report it made in
    the middle, which is what reading forwards or taking a maximum would do. Backwards
    is also the cheaper direction -- the newest report is near the end of a log whose
    per-token deltas written after it.

    A report naming another Turn is skipped rather than trusted. A Session's log holds
    every Turn it has ever run, and a previous Turn's final report can sit arbitrarily
    close to the end of it.
    """
    for record in reversed(events):
        if record.type != turn.TURN_PROGRESS:
            continue
        payload = record.payload
        if str(payload.get("turn_id")) != str(turn_id):
            continue
        idle = payload.get("idle_ms")
        if isinstance(idle, int):
            return idle
        # A report whose field is missing or not a number is not a report this can act
        # on, and it must not fall through to the next-newest one: doing so would let a
        # malformed recent report be answered by a stale healthy one, or by a stale
        # stuck one. Unreadable is treated as unreported, which is the direction that
        # keeps a Turn alive.
        return None
    return None


def latest_report_seq(events: Sequence[EventRecord], turn_id: TurnId) -> Seq | None:
    """The sequence of this Turn's newest progress report, or None if it made none.

    A position and not a time, because the log port hands back no timestamp. The sweep
    compares this against what it saw last pass: a sequence that has moved is a shim
    still talking, and one that has not is a shim that has said nothing since. Elapsed
    time then comes from the sweep's own clock rather than from the log.

    **None is the load-bearing answer.** It means this Turn has never reported, which is
    the permanent condition of every pod running an image built before the emitter, and
    it must never be read as silence -- ceasing and never having begun are different
    facts, and only the first is evidence. Returning a sentinel sequence for both would
    close every one of those pods the day this shipped.

    Scanned backwards for the reason `latest_idle_ms` gives, and matched on `turn_id`
    for the same one: a Session's log holds every Turn it has ever run, and a previous
    Turn's final report can sit arbitrarily close to the end of it.
    """
    for record in reversed(events):
        if record.type != turn.TURN_PROGRESS:
            continue
        if str(record.payload.get("turn_id")) == str(turn_id):
            return record.seq
    return None


def _latest_boundary_per_session(rows: Sequence[object]) -> dict[SessionId, str]:
    """The type of each Session's highest-sequenced turn-boundary row in the window.

    The prescreen's whole content, kept pure so it is gradeable without a log. A Session
    whose latest boundary is a `turn.submitted` has a Turn that nothing has closed since
    -- the three types are strictly ordered within a Session by sequence, and a Turn's
    terminal event always lands above its submission -- so that Session is worth a
    fold.
    One whose latest boundary is a terminal has nothing open and is skipped, which is
    what keeps this sweep from reading the whole log of every Session that ran today.

    A row missing any of the three fields is refused rather than skipped, for the reason
    the neighbouring scan refuses one: a row shape this cannot read would silently empty
    the candidate set, and an empty candidate set reads as "no Session is wedged" --
    exactly the answer that makes this sweep do nothing while doing it quietly.
    """
    latest: dict[SessionId, tuple[int, str]] = {}
    for row in rows:
        session_id = getattr(row, "session_id", None)
        seq = getattr(row, "seq", None)
        type_ = getattr(row, "type", None)
        if session_id is None or seq is None or type_ is None:
            raise TypeError(
                "a turn-boundary row is missing session_id, seq or type: "
                f"{type(row).__name__}"
            )
        key = SessionId(session_id)
        if key not in latest or latest[key][0] < int(seq):
            latest[key] = (int(seq), str(type_))
    return {session: type_ for session, (_, type_) in latest.items()}


class AbandonedTurnSweeper:
    """Closes the open Turns a dead control plane left behind, once per call.

    The collaborators arrive as constructor arguments rather than being reached through
    a `Platform`, which is how `SessionPodReaper` and `FirstTurnPlacement` are both
    written: a collaborator that read fields of the object it is a field of would make
    construction order load-bearing.

    **Holds two counts between sweeps, and either can only delay a close.**
    `_missing_since` remembers, per Session, the Turn whose pod was first seen gone and
    when. That is what makes "gone for two minutes" mean two observed minutes rather
    than one unlucky read, and it is safe to hold in a process that can die because
    losing it costs another two minutes of waiting and can never close a Turn early.
    `_unplaced_since` is the same shape for the other question -- when this Turn was
    first seen with no pod and no `turn.started` -- and holds the same property for the
    same reason. `_silent_since` is the third, and carries a sequence as well as a Turn:
    the question it answers is not "since when has this been true" but "since when has
    this *not moved*", and the position it has not moved from is half of that.

    They are separate dictionaries rather than one because they answer questions that
    happen to share a shape. A Turn is only ever counted by one of them at a time (the
    presence of `turn.started` decides which), but merging them would make a count
    started under one question readable as an answer to the other the moment a Turn
    crossed between them -- and that crossing is exactly what a slow placement is.
    """

    def __init__(
        self,
        *,
        pods: PlacedPods,
        scan: TurnBoundaryScan,
        events: EventLogRange,
        log: EventLogAppend,
        clock: Clock,
    ) -> None:
        self._pods = pods
        self._scan = scan
        self._events = events
        self._log = log
        self._clock = clock
        self._missing_since: dict[SessionId, tuple[TurnId, int]] = {}
        self._unplaced_since: dict[SessionId, tuple[TurnId, int]] = {}
        self._silent_since: dict[SessionId, tuple[TurnId, Seq, int]] = {}

    async def sweep(self) -> Sequence[TurnOutcome]:
        """Decide about every Turn that looks open, and act on each.

        Returns one outcome per open Turn, the untouched ones included, because a sweep
        reporting only its closes cannot be told apart from one that found nothing and
        one that refused everything -- three different operational situations.

        Both windows are read once, before the loop, rather than per Session. One query
        each answers them for every candidate, and reading them per Session would also
        let the windows slide across the sweep, so two Turns judged seconds apart were
        judged against different clocks.

        What the loop then reads per Session is bounded by the Session's Turn count and
        not by its Turns' length: `_SWEEP_READS` names the four types folded here, and
        `latest_progress_of` returns the single newest report rather than the commentary
        it ends. This used to page the whole log, which made judging a Turn cost a row
        for every delta that Turn had ever emitted -- backwards, since the Turns worth
        catching are the long ones. Measured on Postgres 17 at 40 Sessions of 3,000
        events: 782.6 ms over 120,000 rows before, 132.4 ms over 120 after, on a pass
        that runs every 30 seconds while every live Turn reports every 30 seconds.

        A Turn whose own verdict raises is recorded as undecided and the sweep goes
        on. A single unreadable Session must not stop the rest of the platform from
        un-wedging: that failure mode is the one this file exists to prevent, arriving
        by another route.
        """
        now = self._clock.now_epoch_ms()
        # A pod the cluster reports GONE does not count as this Turn's pod being there,
        # and reading the listing without that filter is what made the fast path nearly
        # unreachable in the case it was written for. `_phase_of` answers GONE for a pod
        # carrying a deletion timestamp and for one in Succeeded, Failed or Unknown --
        # and a pod that ran to completion after its control plane died is exactly the
        # second of those. Nothing deletes that object, so it stays in the listing for
        # ever, and counting it as present would reset the grace on every sweep and
        # leave the hour-long ceiling as the only thing that ever closed the Turn.
        #
        # `_candidates` deliberately does NOT filter this way: a Session whose only pod
        # is GONE is precisely one worth folding, and dropping it there would remove it
        # from the sweep altogether rather than merely from this question.
        with_a_pod = {
            pod.session_id
            for pod in await self._pods.placed_pods()
            if pod.phase is not PodPhase.GONE
        }
        candidates = await self._candidates(now)
        outcomes: list[TurnOutcome] = []
        for session_id in sorted(candidates):
            try:
                boundaries = await self._scan.turn_boundaries_of(
                    session_id, _SWEEP_READS
                )
                turn_id = open_turn(boundaries)
                if turn_id is None:
                    self._missing_since.pop(session_id, None)
                    self._unplaced_since.pop(session_id, None)
                    continue
                verdict = await self._decide(
                    session_id,
                    turn_id,
                    boundaries,
                    session_id in with_a_pod,
                    now,
                )
            except Exception:
                # Logged with the traceback rather than swallowed, and recorded as a
                # verdict rather than dropped: a Turn this sweep could not judge is a
                # Turn still wedging its Session, and a caller counting outcomes has to
                # be able to see that.
                _LOG.exception(
                    "the sweep could not decide about the open turn of session %s",
                    session_id,
                )
                outcomes.append(
                    TurnOutcome(
                        session_id=session_id,
                        turn_id=_UNKNOWN_TURN,
                        verdict=TurnVerdict.THE_SWEEP_COULD_NOT_DECIDE,
                    )
                )
                continue
            outcomes.append(
                TurnOutcome(session_id=session_id, turn_id=turn_id, verdict=verdict)
            )
        # Every count for a Session this pass did not consider is dropped, which is what
        # bounds the memory by the live candidate set rather than by everything this
        # process has ever seen. A Session leaves that set by having its Turn closed, by
        # ageing out of the scan window, or by having its log expire -- and in all three
        # a kept count could only be inherited by some later Turn, which is precisely
        # what `_first_seen_missing` refuses on the Turn identifier anyway.
        self._missing_since = {
            session_id: seen
            for session_id, seen in self._missing_since.items()
            if session_id in candidates
        }
        self._unplaced_since = {
            session_id: seen
            for session_id, seen in self._unplaced_since.items()
            if session_id in candidates
        }
        self._silent_since = {
            session_id: seen
            for session_id, seen in self._silent_since.items()
            if session_id in candidates
        }
        return outcomes

    async def _candidates(self, now_ms: int) -> frozenset[SessionId]:
        """Every Session that may have a Turn open, from the log and from the cluster.

        Two sources, unioned, because neither alone is complete. The log's prescreen
        finds a Session whose Turn is open and whose pod is gone, which is the whole
        point of the pod signal and is invisible to a cluster listing. The cluster
        listing finds a Session whose Turn is older than the log window, which the
        prescreen has stopped returning -- and a Turn that old with a pod still up is
        exactly what the ceiling is for.
        """
        latest = _latest_boundary_per_session(
            await self._scan.lifecycle_events_between(
                _TURN_BOUNDARIES, now_ms - _OPEN_TURN_LOOKBACK_MS, now_ms
            )
        )
        from_the_log = {
            session_id
            for session_id, type_ in latest.items()
            if type_ == turn.TURN_SUBMITTED
        }
        from_the_cluster = {pod.session_id for pod in await self._pods.placed_pods()}
        return frozenset(from_the_log | from_the_cluster)

    async def _decide(
        self,
        session_id: SessionId,
        turn_id: TurnId,
        boundaries: Sequence[EventRecord],
        has_pod: bool,
        now_ms: int,
    ) -> TurnVerdict:
        """The whole decision table for one open Turn, in the order the guards run.

        Ordered so that every branch which keeps the Turn is asked before any that
        closes it, and within those, the cluster's own answer before the clock's. A Turn
        whose pod is still there is kept whatever its age says, because a pod at this
        Session's name while this Turn is open is this Turn's pod.

        The first chain splits three ways on facts that are exhaustive and mutually
        exclusive -- the pod is here, or it is not and answered once, or it is not and
        never did -- so each of the two counts is advanced by exactly one arm and
        cleared by the others. That is what keeps a Turn from spending the same wait
        twice under two different questions as it moves between them, which is what a
        slow placement followed by a dying pod actually looks like.
        """
        if not has_pod:
            # A Turn cannot be silent on a pod that is not there -- an absent pod is
            # signal 1's question, and reports stopping because the pod stopped is that
            # signal's evidence, not this one's. Dropping the count means a pod that
            # flickers pays for its silence again from zero, which delays a close and
            # can never advance one: the property both counts above hold.
            self._silent_since.pop(session_id, None)
        if has_pod:
            # The absence has to be continuous, so a pod that is back clears the count
            # rather than leaving it to be resumed by a later absence. Two unlucky reads
            # a day apart must not add up to a close.
            self._missing_since.pop(session_id, None)
        elif started(boundaries, turn_id):
            gone_since = self._first_seen_missing(session_id, turn_id, now_ms)
            if now_ms - gone_since >= POD_GONE_GRACE_MS:
                await close_abandoned_turn(
                    session_id,
                    turn_id,
                    self._log,
                    self._events,
                    turn.TurnFailureCause.RUNTIME_LOST,
                )
                self._missing_since.pop(session_id, None)
                return TurnVerdict.ITS_POD_IS_GONE
        else:
            # No pod and no `turn.started`: this Turn has never been placed. The missing
            # pod is not evidence on its own -- a cold placement was measured holding
            # 660 seconds and looks exactly like this throughout -- so signal 1's count
            # is dropped and the wait is counted separately against a deadline sized for
            # that measurement rather than against signal 1's two-minute grace.
            self._missing_since.pop(session_id, None)
            waiting_since = self._first_seen_unplaced(session_id, turn_id, now_ms)
            if now_ms - waiting_since >= PLACEMENT_DEADLINE_MS:
                await close_abandoned_turn(
                    session_id,
                    turn_id,
                    self._log,
                    self._events,
                    turn.TurnFailureCause.RUNTIME_DID_NOT_START,
                )
                self._unplaced_since.pop(session_id, None)
                return TurnVerdict.IT_NEVER_GOT_A_POD
        if has_pod or started(boundaries, turn_id):
            # Either half clears it: a pod that arrived answers the question this count
            # was asking, and so does a `turn.started`, which proves one arrived even if
            # this sweep is reading the cluster after it went again. Leaving the count
            # standing in either case would let waiting a Turn already survived be spent
            # a second time on a failure that is signal 1's to judge, and reported under
            # a cause that says the runtime never started.
            self._unplaced_since.pop(session_id, None)
        if has_pod:
            # The pod is there, so the pod-gone signal has nothing to say and since the
            # ceiling was removed nothing else does either. This is the only branch from
            # which a wedge is visible, and the only thing that can see it is what the
            # pod says about itself.
            # Fetched here rather than beside the boundaries, because this is the
            # only branch that reads it: a Turn with no pod reaches a verdict without
            # ever asking what its reports said, and paying for that read on every
            # candidate would put back a slice of what bounding the read just removed.
            reports = await self._scan.latest_progress_of(
                session_id, turn.TURN_PROGRESS, str(turn_id)
            )
            idle_ms = latest_idle_ms(reports, turn_id)
            if idle_ms is not None and idle_ms >= STUCK_IDLE_MS:
                await close_abandoned_turn(
                    session_id,
                    turn_id,
                    self._log,
                    self._events,
                    turn.TurnFailureCause.RUNTIME_LOST,
                )
                return TurnVerdict.ITS_RUNTIME_STOPPED_TALKING
            # And if the shim itself died, no report arrives to carry an `idle_ms` at
            # all, so the check above reads whatever the last living report said --
            # healthy, for ever. `None` here is a Turn that never reported and is left
            # alone: ceasing is evidence and never having begun is not.
            report_seq = latest_report_seq(reports, turn_id)
            if report_seq is not None:
                silent_since = self._first_seen_silent(
                    session_id, turn_id, report_seq, now_ms
                )
                if now_ms - silent_since >= REPORTS_CEASED_MS:
                    await close_abandoned_turn(
                        session_id,
                        turn_id,
                        self._log,
                        self._events,
                        turn.TurnFailureCause.RUNTIME_LOST,
                    )
                    self._silent_since.pop(session_id, None)
                    return TurnVerdict.ITS_REPORTS_STOPPED_ARRIVING
            return TurnVerdict.THE_POD_IS_STILL_THERE
        if started(boundaries, turn_id):
            return TurnVerdict.THE_POD_HAS_ONLY_JUST_GONE
        return TurnVerdict.THE_POD_WAS_NEVER_PLACED

    def _first_seen_unplaced(
        self, session_id: SessionId, turn_id: TurnId, now_ms: int
    ) -> int:
        """When this Turn was first seen unplaced, starting the count if it is new.

        Keyed on the Turn for the reason `_first_seen_missing` gives: a Session runs one
        Turn after another and a count inherited across that boundary would spend an
        earlier Turn's waiting on a later Turn's placement.
        """
        remembered = self._unplaced_since.get(session_id)
        if remembered is None or remembered[0] != turn_id:
            self._unplaced_since[session_id] = (turn_id, now_ms)
            return now_ms
        return remembered[1]

    def _first_seen_silent(
        self, session_id: SessionId, turn_id: TurnId, at_seq: Seq, now_ms: int
    ) -> int:
        """When this Turn was first seen stuck at this report, starting a new count.

        Keyed on the sequence as well as the Turn, which is what makes a report arriving
        between two sweeps restart the count rather than merely refresh it. Without the
        sequence in the key this would time a Turn from the first sweep that saw it
        reporting at all, and close every long agent run on the platform -- the retired
        age ceiling's mistake reached through a different door.
        """
        remembered = self._silent_since.get(session_id)
        if remembered is None or remembered[0] != turn_id or remembered[1] != at_seq:
            self._silent_since[session_id] = (turn_id, at_seq, now_ms)
            return now_ms
        return remembered[2]

    def _first_seen_missing(
        self, session_id: SessionId, turn_id: TurnId, now_ms: int
    ) -> int:
        """When this Turn's pod was first seen gone, starting the count if it is new.

        Keyed on the Turn and not only the Session, so a count kept for one Turn is
        never inherited by the next one to run on that Session. That is the identity
        this decision is actually about: the pod object is a lease held for one Turn,
        and the Turn is the durable name of that lease.
        """
        remembered = self._missing_since.get(session_id)
        if remembered is None or remembered[0] != turn_id:
            self._missing_since[session_id] = (turn_id, now_ms)
            return now_ms
        return remembered[1]
