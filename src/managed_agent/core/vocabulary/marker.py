"""The marker family: the event that says a stretch of this log is no longer current.

A marker is not a failure notice and not an error. It says only that work already
recorded was thrown away, and it is the reader -- not the writer -- who decides what to
do about that, which is why the whole judgement lives in the payload's cause rather than
in the type name.

One type rather than one per cause. Every reader handles all of them the same way first
-- leave out the stretch a marker declares -- and only then cares why. A type per cause
would push that branch up into the reader's event dispatch, where a cause added later
arrives as an unrecognized type rather than as an unfamiliar value.
"""

from managed_agent.core.vocabulary import declare

FAMILY = "marker"

WORK_DISCARDED = declare("marker.work_discarded", FAMILY)
