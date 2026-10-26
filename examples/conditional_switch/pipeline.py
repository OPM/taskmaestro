"""Example: a multi-way switch with ``SwitchTask``, ``otherwise()`` and ``first_of``.

``ChooseShipping`` returns one of four labels. Each shipping task is guarded by
the labels it handles; ``otherwise()`` catches every label that no other task
claims (here ``standard`` and ``economy``). ``first_of`` joins the branches.

                                   ┌─ express ──── ExpressShipping ──┐
    LoadParcel ── ChooseShipping ◇─┼─ freight ──── FreightShipping ──┼─ first_of ── PrintLabel
                                   └─ otherwise ── StandardShipping ─┘

Run:
    python examples/conditional_switch/pipeline.py
    python examples/conditional_switch/pipeline.py --weight 120
    python examples/conditional_switch/pipeline.py --priority
    python examples/conditional_switch/pipeline.py --yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel

from taskmaestro import (
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    Runner,
    SwitchTask,
    Task,
    TaskStatus,
    Workflow,
    first_of,
)
from taskmaestro.hooks import LoggingHook

ShippingMethod = Literal["express", "freight", "standard", "economy"]

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ParcelInput(BaseModel):
    """Parcel supplied as job configuration."""

    parcel_id: str
    weight_kg: float
    priority: bool = False
    destination: str


class Parcel(BaseModel):
    parcel_id: str
    weight_kg: float
    priority: bool
    destination: str


class Shipment(BaseModel):
    parcel_id: str
    carrier: str
    days: int
    cost: float


class LabelInput(BaseModel):
    parcel: Parcel
    shipment: Shipment


class ShippingLabel(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class LoadParcel(Task[ParcelInput, Parcel]):
    name = "load_parcel"

    def run(self, input: ParcelInput, ctx: ExecutionContext) -> Parcel:
        return Parcel(**input.model_dump())


class ChooseShipping(SwitchTask[Parcel, ShippingMethod]):
    """The 'switch': returns the label of the branch to run."""

    def decide(self, input: Parcel, ctx: ExecutionContext) -> ShippingMethod:
        if input.weight_kg > 100:
            return "freight"
        if input.priority:
            return "express"
        if input.weight_kg < 1:
            return "economy"
        return "standard"


class ExpressShipping(Task[Parcel, Shipment]):
    def run(self, input: Parcel, ctx: ExecutionContext) -> Shipment:
        return Shipment(parcel_id=input.parcel_id, carrier="AirFast", days=1, cost=49.0)


class FreightShipping(Task[Parcel, Shipment]):
    def run(self, input: Parcel, ctx: ExecutionContext) -> Shipment:
        cost = 2.5 * input.weight_kg
        return Shipment(parcel_id=input.parcel_id, carrier="HeavyHaul", days=7, cost=cost)


class StandardShipping(Task[Parcel, Shipment]):
    def run(self, input: Parcel, ctx: ExecutionContext) -> Shipment:
        return Shipment(parcel_id=input.parcel_id, carrier="PostNord", days=3, cost=9.0)


class PrintLabel(Task[LabelInput, ShippingLabel]):
    def run(self, input: LabelInput, ctx: ExecutionContext) -> ShippingLabel:
        shipment = input.shipment
        return ShippingLabel(
            text=(
                f"{input.parcel.parcel_id} -> {input.parcel.destination} via "
                f"{shipment.carrier} ({shipment.days} days, {shipment.cost:.2f} EUR)"
            )
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def build_workflow() -> Workflow:
    builder = Workflow.builder("conditional_switch")
    parcel = builder.task(
        LoadParcel, config_fields=["parcel_id", "weight_kg", "priority", "destination"]
    )
    method = builder.task(ChooseShipping, depends_on=parcel)
    express = builder.task(ExpressShipping, depends_on=parcel, when=method.on("express"))
    freight = builder.task(FreightShipping, depends_on=parcel, when=method.on("freight"))
    standard = builder.task(StandardShipping, depends_on=parcel, when=method.otherwise())
    builder.task(PrintLabel, parcel=parcel, shipment=first_of(express, freight, standard))
    return builder.build()


def job_configuration(weight_kg: float, priority: bool) -> JobConfiguration:
    return JobConfiguration(
        {
            "load_parcel": {
                "parcel_id": "P-42",
                "weight_kg": weight_kg,
                "priority": priority,
                "destination": "Bergen",
            }
        }
    )


def print_result(result: Job[Any], workflow: Workflow) -> None:
    print("=" * 60)
    print("Conditional switch")
    print("=" * 60)
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
        return
    for task_result in result.task_results:
        marker = "skipped" if task_result.status == TaskStatus.SKIPPED else "ran"
        print(f"  {task_result.task_name:18s} {marker}")
    label = cast(ShippingLabel, result.result)
    print(f"Label: {label.text}")
    print("\nMermaid diagram:")
    print("```mermaid")
    print(workflow.to_mermaid(), end="")
    print("```")


def run_python_mode(weight_kg: float, priority: bool) -> None:
    workflow = build_workflow()
    result = Runner(hooks=[LoggingHook()]).run(
        Job(workflow, EmptyConfig(), job_configuration=job_configuration(weight_kg, priority))
    )
    print_result(result, workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    from taskmaestro.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    print_result(loaded.run(), loaded.workflow)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    example_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Conditional switch example")
    parser.add_argument("--weight", type=float, default=4.0, help="Parcel weight in kg")
    parser.add_argument("--priority", action="store_true", help="Request priority shipping")
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
        run_python_mode(args.weight, args.priority)


if __name__ == "__main__":
    main()
