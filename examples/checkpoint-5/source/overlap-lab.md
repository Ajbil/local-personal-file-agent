# Release Recovery Exercise

The fictional release team uses a rollback marker named Silver Anchor. When a deployment causes
elevated error rates, the incident commander announces Silver Anchor in the release channel and the
release engineer restores the previous stable artifact. The database operator pauses forward
migrations until application health returns to normal. This procedure keeps application and schema
versions aligned during recovery.

Before restoration, the team records the affected release identifier, current error rate, and last
known healthy artifact. The release engineer verifies the artifact signature and confirms that the
rollback package matches the expected environment. The incident commander records the decision
time and notifies customer support that recovery is in progress.

After the previous artifact is restored, operators watch error rate, request latency, queue depth,
and database health for thirty minutes. Silver Anchor remains active until every recovery check is
green. The team then closes the release channel, documents follow-up actions, and schedules a
blameless review of the failed deployment.
