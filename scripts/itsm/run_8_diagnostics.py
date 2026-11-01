"""Step 8 diagnostics: E0-base dev (A) and GRPO train within-group (B). Sampling only.

    # offline, no network, no GPU: the fixed plan and every reservation number
    scripts/itsm/itsm_python.sh scripts/itsm/run_8_diagnostics.py plan
    # cloud only, after separate approval (launched by scripts/itsm/run_8.sh):
    scripts/itsm/itsm_python.sh scripts/itsm/run_8_diagnostics.py run --model DIR --out DIR --confirm-billing [--parts A,B]

``run`` refuses without ``--confirm-billing``, without ``DEEPSEEK_API_KEY`` and without a fixed
hash seed; it runs the dependency preflight first. Budget, GPU time and concurrency rules are
those of configs/itsm/step8/plan.json (see agent_rl.rollout.itsm_diagnostic_runner).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_rl.rollout.itsm_step8 import build_plan, hard_cap_table, load_plan, load_prices, max_system_prompt_bytes, reservation_table  # noqa: E402

#: evaluated in `plan` only; NOT enforced by the current code (a proposal, see the step-8 proposal §3)
PROPOSED_REQUEST_BYTE_CAPS = {"simulator": 65536, "text_judge": 65536, "nl_judge": 262144}


def cmd_plan(args) -> int:
    plan, prices = load_plan(), load_prices()
    groups = build_plan(plan, prices)
    system_bytes = max_system_prompt_bytes(plan)
    print(json.dumps({"reservations": reservation_table(plan, prices),
                      "complete_group_hard_cap": {"current_rules": hard_cap_table(plan, prices, system_bytes=system_bytes),
                                                  "with_proposed_request_size_rule": hard_cap_table(plan, prices, system_bytes=system_bytes,
                                                                                                    request_byte_caps=PROPOSED_REQUEST_BYTE_CAPS)},
                      "groups": [{"group_id": g.group_id, "part": g.part, "samples": g.samples, "envelope_usd": g.envelope_usd} for g in groups]}, indent=1))
    return 0


def cmd_run(args) -> int:
    from agent_rl.utils.hash_seed import require_fixed_hash_seed

    require_fixed_hash_seed("run_8_diagnostics.py")
    if not args.confirm_billing:
        raise SystemExit("refusing to send paid requests without --confirm-billing")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    from agent_rl.data.itsm_runtime_deps import verify_runtime_dependencies

    verify_runtime_dependencies()
    from transformers import AutoTokenizer

    from agent_rl.rollout.itsm_diagnostic_runner import ConcurrencyRamp, DiagnosticRunner
    from agent_rl.rollout.itsm_step8 import make_step8_loop
    from agent_rl.rollout.vllm_token_server import VllmTokenServer
    from agent_rl.utils.budget_manager import BudgetManager, BudgetServer
    from agent_rl.utils.spend_ledger import SpendLedger

    plan, prices = load_plan(), load_prices()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = tuple(p for p in args.parts.split(",") if p)
    groups = build_plan(plan, prices, parts)
    boot = float(os.environ["BOOT_EPOCH"]) - float(os.environ.get("PRIOR_GPU_S", "0"))
    cap_s = plan["gpu"]["cap_gpu_h"] * 3600
    # wall clock -> monotonic deadlines (soft: stop starting groups; hard: cancel, leaving the shutdown margin)
    now_wall, now_mono = time.time(), time.monotonic()
    hard = now_mono + (boot + cap_s - plan["gpu"]["shutdown_margin_s"] - now_wall)
    soft = hard
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    b = plan["budget"]
    with SpendLedger(out / "ledger.jsonl", cap_usd=b["cap_usd"], max_billed_requests=b["max_billed_requests"], budget_id=b["budget_id"]) as ledger:
        manager = BudgetManager(ledger, prices)
        server_ipc = BudgetServer(manager, out / "ipc")
        address, authkey = server_ipc.client_config()

        async def go():
            server = VllmTokenServer.build(args.model, max_model_len=plan["max_model_len"])
            try:
                runner = DiagnosticRunner(
                    plans=groups, budget=manager, sampling_params=plan["sampling"], out_dir=out, max_attempts=plan["max_attempts"],
                    make_loop=lambda g: make_step8_loop(g, server=server, tokenizer=tokenizer, plan=plan, budget_address=address, budget_authkey=authkey),
                    ramp=ConcurrencyRamp(levels=tuple(plan["ramp"]["levels"]), promote_after=plan["ramp"]["promote_after"],
                                         engine_error_stop=plan["ramp"]["engine_error_stop"]),
                    soft_deadline=soft, hard_deadline=hard, group_estimate_s=plan["gpu"]["group_estimate_s"],
                    discard_stop_fraction=plan["discard_stop"]["fraction"], discard_stop_min_groups=plan["discard_stop"]["min_groups"])
                return await runner.run()
            finally:
                server.shutdown()

        try:
            summary = asyncio.run(go())
        finally:
            server_ipc.close()
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary["stop_reason"] is None else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--parts", default="A,B")
    run.add_argument("--confirm-billing", action="store_true")
    args = parser.parse_args()
    for key in ("OPENAI_API_KEY", "MICU_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.pop(key, None)
    return cmd_plan(args) if args.cmd == "plan" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
