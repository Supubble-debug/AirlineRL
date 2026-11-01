"""Fixed Python hash seed for ITSM runs (protocol §13.1, decided after step 7B).

Bench2's ``slice_db_to_orgs`` rebuilds an org-sliced DB by iterating ``set``s of
record ids, so the ORDER (not the content) of unfiltered list results depends on
the interpreter's string-hash seed, which is fixed only at interpreter start.
Every ITSM process that builds an environment must therefore run with
``PYTHONHASHSEED=0`` set BEFORE Python starts:

* shell launchers export it (``scripts/itsm/itsm_python.sh``, ``run_7b.sh``);
* ``ItsmEnvWorker`` children inherit the parent's environment at spawn time and
  report their own ``sys.flags.hash_randomization``; ``ItsmAgentLoop`` refuses a
  child that is not fixed (startup failure, before any sampling);
* verl's Ray workers (step 9) get it through ``runtime_env.env_vars``.

Setting ``os.environ["PYTHONHASHSEED"]`` inside a running interpreter does not
change that interpreter's hashing; it only affects processes started later.
A fixed hash seed makes list-observation order reproducible; it does NOT make
model sampling, the user simulator or vLLM numerics deterministic.
"""

from __future__ import annotations

import os
import sys

REQUIRED_PYTHONHASHSEED = "0"


def hash_seed_report() -> dict:
    """What this interpreter was started with (``hash_randomization == 0`` <=> seed 0)."""
    return {
        "pid": os.getpid(),
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "hash_randomization": sys.flags.hash_randomization,
        "fixed": sys.flags.hash_randomization == 0,
    }


def interpreter_hash_seed_fixed() -> bool:
    # sys.flags is set at startup; PYTHONHASHSEED=0 is the only setting that disables randomization.
    return sys.flags.hash_randomization == 0


def require_fixed_hash_seed(where: str) -> None:
    if not interpreter_hash_seed_fixed():
        raise RuntimeError(
            f"{where}: this interpreter was started without PYTHONHASHSEED={REQUIRED_PYTHONHASHSEED} "
            f"(env PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')!r}); launch it through "
            "scripts/itsm/itsm_python.sh or export PYTHONHASHSEED=0 before starting Python"
        )


def reexec_with_fixed_hash_seed() -> None:
    """For deterministic CLI builders: restart the interpreter with the seed set (before any work)."""
    if not interpreter_hash_seed_fixed():
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, "PYTHONHASHSEED": REQUIRED_PYTHONHASHSEED})
