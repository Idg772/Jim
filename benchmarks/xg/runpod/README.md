# Runpod execution for the frozen CE XG run

This workflow packages committed `HEAD` rather than the working tree, overlays
the verified CE PSD and both pinned DE405 ephemeris files, and checks every
archive member against a content manifest. It then runs qualification, manifest
assembly, and sampling in one immutable pod environment. Any failed receipt,
manifest mismatch, source mutation, hardware mismatch, test failure, sampler
failure, or artifact mismatch prevents a successful result.

The package command refuses to proceed until the production qualification
entrypoint is committed at `benchmarks/xg/run_qualification.py`. It invokes that
entrypoint as follows:

```console
python benchmarks/xg/run_qualification.py \
  --config benchmarks/xg/xg-ce-4096-65536.toml \
  --bundle-dir RESULT/qualification \
  --n-devices 4
```

Prepare the pinned PSD and DE405 ephemerides, commit the complete XG
implementation, and package it:

```console
uv run python benchmarks/xg/prepare_inputs.py
bash benchmarks/xg/runpod/package_workspace.sh
```

Inspect the immutable four-H200 launch without creating a pod:

```console
bash benchmarks/xg/runpod/provision.sh
```

The launcher has no GPU, image, or deadline override. With `--launch`, it checks
the live Secure Cloud price, account balance, and absence of another paid
resource before creating anything. It refuses a projected H200 charge above
$40 or a remaining balance below $10. Runpod will delete the pod after exactly
120 minutes even if the local controller disappears.

After inspection, create the pod and capture its launch receipt. The spending
path refuses to launch until the exact package that will be uploaded verifies
locally:

```console
bash benchmarks/xg/runpod/provision.sh --launch \
  --archive /private/tmp/jim-xg-REVISION.tar.gz
```

The launcher writes a no-overwrite launch receipt that binds the pod, package,
pinned image, four-H200 allocation, and exact deletion deadline. Upload and
execute the printed receipt and package:

```console
python benchmarks/xg/runpod/upload_and_run.py \
  /private/tmp/jim-xg-REVISION.tar.gz.runpod-launch.json \
  /private/tmp/jim-xg-REVISION.tar.gz
```

Before SSH, the controller queries the live pod and requires the exact image,
GPU type/count, Secure Cloud placement, running state, workspace mount, and SSH
port. It then pins the first SSH host key, verifies the uploaded SHA-256, refuses
existing remote paths, and downloads a content-addressed result archive for
both success and scientific failure. It deletes the pod only after that archive
verifies locally. `--keep-pod` is available for an explicitly requested post-run
inspection; the automatic deletion deadline remains in force.

The remote process reserves at most 55 minutes for qualification, five minutes
for manifest assembly, and 30 minutes for sampling. Setup, tests, packaging, and
retrieval use the remaining deadline. It installs all dependencies before
assembly and performs no package synchronization or source edits between
assembly and the science run, because the qualification manifest binds the
exact Python and platform runtime.
