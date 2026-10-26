"""Example: an if/else branch with ``ConditionTask`` and ``first_of``.

Large orders are sent to manual review, the rest are approved automatically.
Exactly one of the two branches runs; the other is skipped. ``first_of`` joins
the branches again by taking the decision of whichever branch ran.

                          ┌─ true ── ManualReview ─┐
    LoadOrder ── IsLarge ◇┤                        ├─ first_of ── Finalize
                          └─ false ─ AutoApprove ──┘

Run:
    python examples/conditional_if/pipeline.py
    python examples/conditional_if/pipeline.py --amount 25000
    python examples/conditional_if/pipeline.py --yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from taskmaestro import (
    ConditionTask,
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    Runner,
    Task,
    TaskStatus,
    Workflow,
    first_of,
)
from taskmaestro.hooks import LoggingHook

REVIEW_THRESHOLD = 10_000.0

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class OrderInput(BaseModel):
    """Order supplied as job configuration."""

    order_id: str
    customer: str
    amount: float


class Order(BaseModel):
    order_id: str
    customer: str
    amount: float


class Decision(BaseModel):
    order_id: str
    approved: bool
    reviewer: str


class FinalizeInput(BaseModel):
    order: Order
    decision: Decision


class Receipt(BaseModel):
    order_id: str
    status: str
    reviewer: str


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class LoadOrder(Task[OrderInput, Order]):
    name = "load_order"

    def run(self, input: OrderInput, ctx: ExecutionContext) -> Order:
        return Order(**input.model_dump())


class IsLargeOrder(ConditionTask[Order]):
    """The 'if': True when the order needs a manual review."""

    def decide(self, input: Order, ctx: ExecutionContext) -> bool:
        return input.amount >= REVIEW_THRESHOLD


class ManualReview(Task[Order, Decision]):
    def run(self, input: Order, ctx: ExecutionContext) -> Decision:
        ctx.logger.info("Order %s needs manual review", input.order_id)
        return Decision(order_id=input.order_id, approved=True, reviewer="finance-team")


class AutoApprove(Task[Order, Decision]):
    def run(self, input: Order, ctx: ExecutionContext) -> Decision:
        return Decision(order_id=input.order_id, approved=True, reviewer="auto")


class Finalize(Task[FinalizeInput, Receipt]):
    def run(self, input: FinalizeInput, ctx: ExecutionContext) -> Receipt:
        status = "approved" if input.decision.approved else "rejected"
        return Receipt(
            order_id=input.order.order_id, status=status, reviewer=input.decision.reviewer
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def build_workflow() -> Workflow:
    builder = Workflow.builder("conditional_if")
    order = builder.task(LoadOrder, config_fields=["order_id", "customer", "amount"])
    is_large = builder.task(IsLargeOrder, depends_on=order)
    manual = builder.task(ManualReview, depends_on=order, when=is_large.if_true())
    auto = builder.task(AutoApprove, depends_on=order, when=is_large.if_false())
    builder.task(Finalize, order=order, decision=first_of(manual, auto))
    return builder.build()


def job_configuration(amount: float) -> JobConfiguration:
    return JobConfiguration(
        {"load_order": {"order_id": "A-1001", "customer": "Acme", "amount": amount}}
    )


def print_result(result: Job[Any], workflow: Workflow) -> None:
    print("=" * 60)
    print("Conditional if/else")
    print("=" * 60)
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
        return
    for task_result in result.task_results:
        marker = "skipped" if task_result.status == TaskStatus.SKIPPED else "ran"
        print(f"  {task_result.task_name:16s} {marker}")
    receipt = cast(Receipt, result.result)
    print(f"Receipt: {receipt.order_id} {receipt.status} by {receipt.reviewer}")
    print("\nMermaid diagram:")
    print("```mermaid")
    print(workflow.to_mermaid(), end="")
    print("```")


def run_python_mode(amount: float) -> None:
    workflow = build_workflow()
    result = Runner(hooks=[LoggingHook()]).run(
        Job(workflow, EmptyConfig(), job_configuration=job_configuration(amount))
    )
    print_result(result, workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    from taskmaestro.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    print_result(loaded.run(), loaded.workflow)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Conditional if/else example")
    parser.add_argument("--amount", type=float, default=250.0, help="Order amount")
    parser.add_argument(
        "--yaml",
        metavar="FILE",
        nargs="?",
        const=str(example_dir / "workflow.yaml"),
        help="Load YAML workflow (default: workflow.yaml)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        default=str(example_dir / "input.yaml"),
        help="Input YAML file (default: input.yaml)",
    )
    args = parser.parse_args()

    if args.yaml:
        run_yaml_mode(args.yaml, args.input)
    else:
        run_python_mode(args.amount)


if __name__ == "__main__":
    main()
