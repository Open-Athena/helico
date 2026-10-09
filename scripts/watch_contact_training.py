"""One-owner, bounded CoreWeave training monitor. Run with the Iris Python env.

Writes a durable local status record, notifies on failures/completion, and can
resume transient worker/storage failures. It never retries scientific/data
failures, cancels other jobs, or changes cluster configuration.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def retryable_failure(text):
    text = text.lower()
    if any(x in text for x in ("nonfinite", "out of memory", "oomkilled", "dataloader", "data loader",
                               "assertionerror", "filenotfounderror", "shape mismatch")):
        return False
    return any(x in text for x in ("worker lost", "node lost", "worker_failed",
                                   "endpointconnectionerror", "readtimeouterror",
                                   "connection reset by peer", "temporarily unavailable"))


def storage(state):
    import fsspec
    command = ["kubectl", "--kubeconfig", state["kubeconfig"], "--context", state["kube_context"],
               "-n", "iris", "get", "secret", "iris-task-env", "-o", "json"]
    secret = json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL, timeout=30))["data"]
    for key in ("FSSPEC_S3", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
        if key in secret:
            os.environ[key] = base64.b64decode(secret[key]).decode()
    options = json.loads(os.environ.get("FSSPEC_S3", "{}"))
    options["endpoint_url"] = "https://cwobject.com"
    options.setdefault("client_kwargs", {}).pop("endpoint_url", None)
    return fsspec.filesystem("s3", skip_instance_cache=True, **options)


def read_json(fs, path):
    if not fs.exists(path):
        return None
    with fs.open(path) as stream:
        return json.load(stream)


def inspect(state):
    from iris.cli.connect import open_iris_client
    from iris.cluster.types import JobName
    with open_iris_client(config_file=Path(state["cluster_config"]), workspace=Path(state["workspace"])) as client:
        status = client.job(JobName.from_wire(state["job_id"])).status()
    snapshot = {"checked_at": time.time(), "job_state": str(status.state),
                "tasks": [str(x.state) for x in status.tasks], "error": status.error_message}
    fs = storage(state)
    base = state["spec"]["output_uri"]
    snapshot["checkpoint"] = read_json(fs, base + "/latest.json")
    snapshot["result"] = read_json(fs, base + "/result.json")
    root = base + "/diagnostics"
    if fs.exists(root):
        attempts = sorted(fs.ls(root, detail=False))
        if attempts:
            attempt = attempts[-1]
            snapshot["diagnostic_uri"] = "s3://" + attempt
            ranks = [read_json(fs, attempt + f"/rank-{rank}-progress.json") for rank in range(state["spec"]["gpus"])]
            snapshot["ranks"] = [r for r in ranks if r]
            snapshot["completed_step"] = min((r.get("completed_step", 0) for r in snapshot["ranks"]), default=0)
            if str(status.state) in {"failed", "worker_failed"}:
                log = attempt + "/trainer.log"
                if fs.exists(log):
                    snapshot["failure_tail"] = fs.cat_file(log, start=max(0, fs.size(log) - 32768)).decode(errors="replace")
    if snapshot["checkpoint"]:
        uri = snapshot["checkpoint"]["checkpoint"]
        snapshot["checkpoint_bytes"] = fs.size(uri) if fs.exists(uri) else 0
    if str(status.state) == "succeeded":
        config = json.loads((Path(state["workspace"]) / state["spec"]["config"]).read_text())
        code = ("import json,sys,wandb; r=wandb.Api(timeout=30).run('timodonnell/helico/'+sys.argv[1]); "
                "print(json.dumps({'state':r.state,'step':r.lastHistoryStep}))")
        raw = subprocess.check_output([str(Path(state["workspace"]) / ".venv/bin/python"),
                                       "-c", code, config["run_name"]],
                                      stderr=subprocess.DEVNULL, timeout=60, text=True)
        snapshot["wandb"] = json.loads(raw)
    return snapshot


def notify(message):
    print(json.dumps({"notification": message, "time": time.time()}), flush=True)
    try:
        subprocess.run(["notify-send", "Helico training", message], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def save(path, state):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(state, indent=2) + "\n")
    pending.replace(path)


def recover(state):
    from helico.experiment import ensure_training_run, set_experiment
    workspace = Path(state["workspace"])
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=workspace, text=True).strip():
        raise RuntimeError("Workspace changed; recovery needs review")
    # A monitor may have been added since submission, but scientific code must
    # still match the inspected attempt. Never silently launch unreviewed code.
    subprocess.run(["git", "diff", "--exit-code", state["spec"]["git_sha"], "HEAD", "--",
                    "src", "configs", "scripts/coreweave_bootstrap.sh", "uv.lock"],
                   cwd=workspace, check=True, stdout=subprocess.DEVNULL)
    remaining = int(state["deadline"] - time.time())
    if remaining < 1800:
        raise RuntimeError("Recovery would exceed the original run deadline")
    number = state["restart_count"] + 1
    spec = {k: v for k, v in state["spec"].items() if k != "git_sha"}
    spec.update(job_name=state["base_job_name"] + f"-recovery-{number}", timeout_seconds=remaining)
    name = state["base_step_name"] + f"-recovery-{number}"
    set_experiment("exp22_contact-scale")
    os.environ.update(HELICO_IRIS_PYTHON=sys.executable, HELICO_IRIS_CONFIG=state["cluster_config"])
    kwargs = dict(gpu=f"H100:{spec['gpus']}", max_steps=20000, crop_size=384, lr=2e-5,
                  est_wall_hours=remaining / 3600, coreweave=spec)
    os.environ["HELICO_DRY_RUN"] = "1"
    ensure_training_run(name, **kwargs)
    del os.environ["HELICO_DRY_RUN"]
    if spec["estimated_incremental_cost_usd"] > 100:
        raise RuntimeError("Whole-experiment cost gate exceeded")
    receipt = ensure_training_run(name, **kwargs).meta
    state.update(job_id=receipt["job_id"], spec=receipt["spec"], restart_count=number)
    notify(f"Resuming {state['job_id']} from the last durable checkpoint.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-recover", action="store_true")
    args = parser.parse_args()
    lock = args.state.with_suffix(".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(args.state.read_text())
    state.update(owner_pid=os.getpid(), auto_recover=args.auto_recover)
    while True:
        delay = 570
        try:
            snapshot = inspect(state)
            state["latest_signal"] = snapshot
            condition = snapshot["job_state"]
            if condition == "succeeded":
                result, checkpoint = snapshot.get("result"), snapshot.get("checkpoint")
                verified = bool(result and result.get("finished_steps") and checkpoint and
                                checkpoint["step"] == result["step"] and snapshot.get("checkpoint_bytes", 0) > 0 and
                                snapshot.get("wandb", {}).get("state") == "finished")
                state["monitor_status"] = "completed" if verified else "needs_attention"
                notify("Training completed with final checkpoint." if verified else "Job exited successfully but final artifacts are incomplete.")
                save(args.state, state)
                return
            if condition in {"failed", "worker_failed", "killed", "unschedulable"}:
                failure = snapshot.get("failure_tail", "") + snapshot["error"] + condition
                can_retry = (args.auto_recover and condition in {"failed", "worker_failed"} and
                             state["restart_count"] < 3 and snapshot.get("checkpoint_bytes", 0) > 0 and
                             retryable_failure(failure))
                if can_retry:
                    recover(state)
                    delay = 120
                else:
                    state["monitor_status"] = "needs_attention"
                    notify(f"{state['job_id']} is {condition}; diagnostics are preserved and automatic retries are stopped.")
                    save(args.state, state)
                    return
            ranks = snapshot.get("ranks", [])
            if ranks and time.time() - max(r["timestamp"] for r in ranks) > 1200:
                state["monitor_status"] = "stalled"
                if not state.get("stall_notified"):
                    notify("Training has made no observable progress for 20 minutes; inspect per-rank diagnostics.")
                    state["stall_notified"] = True
            else:
                state["monitor_status"] = "watching"
                state["stall_notified"] = False
            state.pop("inspection_error", None)
            state["inspection_failures"] = 0
        except Exception as error:
            # An unavailable control plane is not proof that training failed.
            state["inspection_error"] = type(error).__name__
            state["monitor_status"] = "inspection_retry"
            state["inspection_failures"] = state.get("inspection_failures", 0) + 1
            if state["inspection_failures"] == 3:
                notify("Training monitoring/recovery could not complete three checks; inspect the saved monitor state.")
            delay = 60
        save(args.state, state)
        if args.once:
            return
        time.sleep(delay)


if __name__ == "__main__":
    main()
