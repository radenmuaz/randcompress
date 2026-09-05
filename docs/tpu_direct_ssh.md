# Direct SSH to a TPU VM (bypassing `gcloud ... ssh` per-call overhead)

**Do this immediately on every fresh TPU node connection** — right after confirming the node is
`READY`, before installing anything or scp'ing the repo — not as an optional later optimization.
`gcloud compute tpus queued-resources ssh <qr-name>` works but re-validates TPU state and
re-preps the node on every single call (~3-25s overhead each time, more if there's a
maintenance/preemption event to detect); every command on the node after this point should go
through the direct connection set up below instead.

Once a TPU VM is up and you've SSH'd into it via gcloud at least once (which propagates your
`~/.ssh/google_compute_engine` public key to the instance), you can talk to it directly with
plain `ssh` and a multiplexed connection.

## 1. Get the actual node name and external IP

The queued-resource name (e.g. `tpu1r`) is not always the underlying node name — gcloud ssh's
own "Finished preparing node <name>." line reveals it (e.g. `tpu1r` queued resource -> node
`tpu1`). Then:

```bash
gcloud compute tpus tpu-vm describe <node-name> --project raden-tpu --zone <zone> \
  --format="yaml(networkEndpoints,state)"
```

gives `networkEndpoints[0].accessConfig.externalIp` and `state` (must be `READY`).

## 2. Plain direct SSH

```bash
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 \
  -i ~/.ssh/google_compute_engine muaz@<external_ip> "echo ok"
```

No OS Login is configured on this project (checked project/instance metadata — no
`enable-oslogin`), so this is plain metadata-based SSH key auth; the key gcloud already
propagated on the first `queued-resources ssh` call keeps working directly.

## 3. Persistent multiplexed connection (skip repeated handshakes)

```bash
mkdir -p ~/.ssh/controlmasters
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ControlMaster=auto -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p \
  -o ControlPersist=6h -i ~/.ssh/google_compute_engine -fN muaz@<external_ip>
```

Then every subsequent command reuses the open connection (~0.3s vs several seconds through
gcloud):

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "<command>"
```

`scp` can use the same `-o ControlPath=...` flag to reuse the connection too.

## 4. Enable linger before using `/dev/shm` for anything long-lived

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> 'sudo loginctl enable-linger muaz'
```

Do this on every fresh node right after step 3, before launching any `tmux`-detached job that
writes to `/dev/shm`. Without it, `systemd-logind`'s default `RemoveIPC=yes` wipes the user's
tmpfs-owned files once their last tracked SSH session ends — a plain `ssh ... 'tmux new-session -d
...'` does NOT keep a session alive from logind's perspective once that SSH call returns, even
though the detached tmux process keeps running. See CLAUDE.md's tmpfs section for the full
incident (silently lost `/dev/shm` training data on tpu1/2/3, 2026-08-28) — this is a one-time,
reboot-persistent fix per node, not a per-command workaround.

## Caveats

- **Every TPU listed in [TPU.md](../TPU.md) is a spot instance** — it can be preempted (reclaimed)
  by Google at any time, with no warning, mid-session, mid-training-run. Direct ssh does **not**
  check TPU state first, so preemption shows up purely as a sudden connectivity failure against a
  node that was working seconds ago: the command might hang until `ConnectTimeout`/TCP timeout,
  or fail immediately with `Connection refused`/`No route to host` (the VM itself is gone, not
  just unresponsive) — both are consistent with preemption, not just "network is flaky". **The
  very first thing to check when a previously-working direct-ssh connection suddenly can't
  connect is TPU state, not connection retries or a new ControlMaster**:
  `gcloud compute tpus queued-resources describe <qr-name> --project raden-tpu --zone <zone> --format="value(state.state)"`
  — if it prints `PREEMPTED`, the node is gone and won't come back; per CLAUDE.md, don't create a
  replacement yourself, report it and ask how to proceed. Only retry the connection if state
  still says `READY`/`ACTIVE`.
- The multiplexed master survives across separate tool calls/shells as long as its process
  (`ControlPersist`) is alive — check with
  `ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -O check muaz@<external_ip>`.
- `pgrep -f`/`pkill -f` run over ssh will self-match the ssh command's own argument string if
  the pattern is a substring of the command you're running it from — use `ps aux | grep -i
  <name>` to eyeball real PIDs first when in doubt.
- **A `gcloud compute tpus queued-resources ssh ... --command="echo ok"` reconnect call (used to
  fix a stale/dropped ControlMaster before re-establishing direct ssh) may kill a lingering `tmux`
  session and everything running inside it, with no error and no trace in the training process's
  own log** — observed directly (2026-08-23): a multi-hour training run in a `tmux` session
  stopped mid-step with no traceback, and by the next check-in the `tmux` server itself was gone
  (`tmux ls` → "no server running"), `ps aux` showed no trace of the training process, `uptime`
  showed no VM reboot — this happened shortly after a `gcloud ... ssh --command="echo ok"`
  reconnect call (its "preparing node" step re-preps the instance, which appears able to reset the
  user session tmux is attached to). Root cause not fully confirmed, but the correlation was
  clean and reproduced the pattern described as "risky" for a `tmux`-hosted long-running job.
  **Mitigation**: prefer reconnecting via the plain SSH commands in this doc (steps 2/3 above)
  over re-running `gcloud ... ssh --command=...` on a node with a long-running `tmux` session, if
  at all avoidable; if a `gcloud ... ssh` call is unavoidable for reconnect, check the `tmux`
  session's actual liveness (`tmux ls`, `ps aux | grep python3`) right after, don't assume it
  survived just because the reconnect itself succeeded.
