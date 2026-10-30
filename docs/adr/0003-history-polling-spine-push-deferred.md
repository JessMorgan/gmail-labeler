# History-API polling is the spine; Pub/Sub push deferred

Gmail offers no true push: Pub/Sub notifications are opaque, best-effort, and
still require the History API to learn what changed — plus a public HTTPS
endpoint (cert, port 443, watch renewal every ≤7 days), which a strictly-local
tool would need a tunnel for anyway. We decided the daemon syncs via
`users.history.list` on a 30s ticker against a persisted `historyId` cursor;
crash recovery comes free (the next poll replays the gap). Push becomes a
later latency optimization that merely *triggers* a sync — the polling design
is the superset, so nothing is redesigned when it lands. An explicit
1–2 minute poll interval is a supported config value for quota-sensitive
users.
