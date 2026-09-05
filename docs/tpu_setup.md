# TPU VM setup

End-to-end steps from a bare TPU VM (see [TPU.md](../TPU.md) for available queued resources —
**never create a new one**, only use nodes already listed) to a running JAX training job. Active
lineages (`gpt2_jax/`, `summformer_jax/`) are pure JAX/Flax — no torch/torch_xla install, no
version pinning, no flash-attention kernel setup needed. **Right after confirming the node is
READY, set up direct ssh per [docs/tpu_direct_ssh.md](tpu_direct_ssh.md) before anything else** —
every step below should go through that persistent connection, not repeated `gcloud ... ssh` calls
(shown here in `gcloud` form only for copy-paste clarity).

## 0. Confirm the node is READY

```bash
gcloud compute tpus queued-resources describe <qr-name> --project raden-tpu --zone <zone> \
  --format="value(state.state)"
```

Queued-resource name (e.g. `tpu1r`) and actual node name (e.g. `tpu1`) can differ — `gcloud ssh`'s
`Finished preparing node <name>.` line reveals the real one.

## 1. Direct ssh (do this immediately)

See [docs/tpu_direct_ssh.md](tpu_direct_ssh.md) in full — one `gcloud ... ssh --command="echo ok"`
to propagate the key, get the external IP, then set up the persistent `ControlMaster` connection.
Every command below should use that connection, not `gcloud ... ssh`.

## 2. Install `uv` and sync the env

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv` fetches its own Python 3.12 (`pyproject.toml` requires `>=3.12`, stock VM ships 3.10).

```bash
source $HOME/.local/bin/env && cd ~/qcute && uv sync
```

`uv sync` installs JAX + deps straight from `pyproject.toml`/`uv.lock` — no separate venv juggling,
no ABI version pinning against a plugin (that was a torch_xla-specific problem, doesn't apply).

## 3. Sanity-check JAX sees the TPU chips

```bash
cd ~/qcute && .venv/bin/python3 -c "import jax; print(jax.devices())"
```

Expect a list of `TpuDevice`s. If this hangs, errors, or shows CPU devices only, stop and fix it
before scp'ing further — most likely cause is TPU preemption (check
`gcloud compute tpus queued-resources describe <qr-name> ... --format="value(state.state)"`).

## 4. scp the project over

```bash
tar czf /tmp/qcute_src.tar.gz \
  --exclude='.venv' --exclude='.git' --exclude='datasets' --exclude='logs' \
  --exclude='checkpoints' --exclude='__pycache__' --exclude='*.pyc' \
  gpt2_jax summformer_jax configs scripts docs pyproject.toml uv.lock CLAUDE.md TPU.md

gcloud compute tpus queued-resources scp /tmp/qcute_src.tar.gz <qr-name>:~/qcute_src.tar.gz \
  --project raden-tpu --zone <zone>

gcloud compute tpus queued-resources ssh <qr-name> --project raden-tpu --zone <zone> \
  --command="cd ~/qcute && tar xzf ~/qcute_src.tar.gz && rm ~/qcute_src.tar.gz && ls"
```

To re-sync after local edits, redo the tar+scp+extract (no rsync-over-gcloud-ssh path set up) —
**or** `scp` individual changed files directly over the direct-ssh connection, cheaper for a
one-file edit.

## 5. Launch training inside `tmux`, never a bare blocking `gcloud ssh --command`

Any long-running or user-monitorable command (installs, data prep, training) goes inside a named
`tmux` session on the VM — decouples the remote process's lifetime from any one SSH connection, so
the user (or a later session) can reattach anytime. `tmux` is preinstalled on the standard image.

**Prefer no stdout redirect** — tqdm/prints land straight in the pane:

```bash
tmux new-session -d -s <run_name> "cd ~/qcute && .venv/bin/python3 -u <module/script.py> --config <config.py> --run-name <run_name>"
```

Only redirect (via `tee`, not a plain `>`) when the run's log needs periodic `scp`-pulling — see
CLAUDE.md's training-run-conventions section for why plain `>` leaves the tmux pane blank.

Give the user the attach/peek commands:

```bash
tmux attach -t <run_name>              # Ctrl-b d to detach without killing
tmux capture-pane -t <run_name> -p -S -40   # peek without attaching
```

## Monitoring a multi-hour run

Check in periodically (~hourly for a multi-hour run):

```bash
ssh -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip> "tmux capture-pane -t <run_name> -p -S -20"
```

- Note the latest metric (val_bpb/val_loss) trend, confirm no traceback.
- **Connection suddenly fails on a previously-working node** → check for preemption first:
  `gcloud compute tpus queued-resources describe <qr-name> ... --format="value(state.state)"`.
  `PREEMPTED` = node and everything on it is gone; report and ask before retrying/replacing.
  Full detail: [docs/tpu_direct_ssh.md](tpu_direct_ssh.md).
- **Pull back only `run.jsonl`/`log.jsonl`, not `run.log`/checkpoints**, to save egress — copy to
  the same relative path under the local repo's `logs/<run_name>/` so `scripts/plot_run.py
  logs/<run_name>` works unedited:

```bash
scp -o ControlPath=~/.ssh/controlmasters/<tag>-%r@%h:%p -i ~/.ssh/google_compute_engine \
  muaz@<external_ip>:~/qcute/logs/<run_name>/log.jsonl logs/<run_name>/log.jsonl
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `uv: command not found` on a later ssh call | each `ssh --command=...` is a fresh non-login shell | `source $HOME/.local/bin/env` at the start of every command |
| `ssh ... exited with return code 255` / hangs | TPU preempted (spot), or transient SSH flakiness | check `queued-resources describe ... state.state`; retry |
| `jax.devices()` shows CPU only | TPU not actually attached, or a stale process still holds it | `ps aux \| grep python3`, kill stale PID; check preemption |
| training silently uses growing-prefix data | dataloader reads whatever shards exist at startup | confirm data prep (`scripts/prep_imagenet64.py` etc.) actually finished before launching |
| `/dev/shm` data vanishes mid-run | `systemd-logind` `RemoveIPC=yes` wiping tmpfs after SSH session ends | `sudo loginctl enable-linger muaz` once per node (see CLAUDE.md) |

## Archived: torch/torch_xla setup (`qcute.bytelm_tpu` lineage, superseded by JAX)

The `qcute.bytelm_tpu` lineage (torch_xla, flash-attention Pallas kernels, `--multichip`
collective training, `zero_kv_sink` cost investigation, FineWeb-Edu byte-level prep) is archived —
no longer the active TPU path. Full history (version-pin ABI mismatches, nightly-build setup, the
`--multichip` hang investigation, batch-size sweeps) preserved in git history of this file if ever
needed again; not repeated here since none of it applies to the active JAX lineages.
