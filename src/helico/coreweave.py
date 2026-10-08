"""Submit durable Iris root jobs; no cluster lifecycle or Kubernetes mutations."""
from __future__ import annotations

import argparse
import json
import netrc
import os
from pathlib import Path


def submit(spec: dict, cluster_config: Path, workspace: Path) -> dict:
    # Iris is an operator-side dependency, deliberately outside the model env.
    from fray.iris_backend import FrayIrisClient
    from fray.types import ResourceConfig, Entrypoint, JobRequest, create_environment
    from iris.cli.connect import open_iris_client
    from rigging.timing import Duration

    env = {"PYTHONUNBUFFERED": "1", "WANDB_PROJECT": "helico",
           "WANDB_ENTITY": "timodonnell", "OMP_NUM_THREADS": "2",
           "HELICO_CODE_SHA": spec["git_sha"]}
    if spec["mode"] == "train":
        key = os.environ.get("WANDB_API_KEY")
        if not key:
            auth = netrc.netrc().authenticators("api.wandb.ai")
            key = auth[2] if auth else None
        if not key:
            raise RuntimeError("W&B credential is required before submission")
        env["WANDB_API_KEY"] = key
    common = dict(cpu=spec["cpu"], ram=spec["memory"], disk=spec["disk"],
                  image=spec["image"], preemptible=False)
    resources = (ResourceConfig.with_gpu("H100", count=spec["gpus"], **common)
                 if spec["gpus"] else ResourceConfig(**common))
    request = JobRequest(
        name=spec["job_name"], resources=resources, priority=3,
        environment=create_environment(workspace=str(workspace),
            env_vars=env, setup_scripts=[]),
        entrypoint=Entrypoint.from_binary("bash", ["scripts/coreweave_bootstrap.sh",
            spec["mode"], spec["config"]]),
        timeout=Duration.from_seconds(spec["timeout_seconds"]),
        max_retries_failure=0, max_retries_preemption=3, max_task_failures=0,
    )
    with open_iris_client(config_file=cluster_config, workspace=workspace) as client:
        handle = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=True)
        return {"job_id": handle.job_id, "cluster": "cw-rno2a", "status": "submitted",
                "spec": spec}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", type=Path)
    p.add_argument("--cluster-config", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = submit(json.loads(args.spec.read_text()), args.cluster_config, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
