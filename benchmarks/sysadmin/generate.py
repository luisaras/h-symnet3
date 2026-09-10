#!/usr/bin/env python3
"""
SysAdmin instance generator (RDDL + PPDDL).

Generates a random SysAdmin network topology and returns/writes:
  - an RDDL instance (non-fluents + instance blocks) pairing with a
    sysadmin_mdp.rddl domain,
  - a matching PPDDL domain + problem, encoded as a proper finite-horizon
    SSP (boolean step-counter goal, no numeric fluents besides the
    built-in reward) so it loads cleanly in SSP solvers such as SSiPP.

Notes on the PPDDL encoding (see prior discussion for full rationale):
  - The PPDDL domain is instance-specific: a computer's transition
    probability depends on the RUNNING COUNT of its neighbours, which
    PPDDL's `probabilistic` effect can't express as a formula, so it is
    compiled into flat (non-nested) `(when (and ...) ...)` clauses, one
    per neighbour-state combination. This is O(2^k) per computer, where
    k = connections per computer -- keep k modest.
  - Horizon is encoded as a boolean step-counter chain (step0..stepH)
    rather than a numeric fluent/comparison, since SSP solvers built on
    the mdpsim PPDDL parser generally only support boolean predicates
    plus the reserved `reward` accumulator, not user-defined numeric
    fluents or numeric preconditions.
  - All object constants (computers + step markers) are declared in a
    single `:constants` clause -- some parsers reject multiple
    `:constants` clauses in one domain.
"""

import itertools
import os
import random
import sys

rng = random.Random()

# --------------------------------------------------------------------------
# Topology
# --------------------------------------------------------------------------

def generate_topology(n, k):
    """Return (computers, incoming) where incoming[c] is the set of
    computers y such that CONNECTED(y, c) holds (y feeds into c).

    Guarantees weak connectivity via a base ring, then fills each
    computer's in-degree up to k (capped at n-1) with random extra edges.
    """
    computers = [f"computer{i + 1}" for i in range(n)]

    incoming = {c: set() for c in computers}
    for i in range(n):
        src = computers[i]
        dst = computers[(i + 1) % n]
        if dst != src:
            incoming[dst].add(src)

    k = max(0, min(k, n - 1))
    for c in computers:
        needed = k - len(incoming[c])
        if needed <= 0:
            continue
        candidates = [x for x in computers if x != c and x not in incoming[c]]
        rng.shuffle(candidates)
        for x in candidates[:needed]:
            incoming[c].add(x)

    return computers, incoming


# --------------------------------------------------------------------------
# RDDL instance
# --------------------------------------------------------------------------

def build_rddl(instance_name, computers, incoming, prob, penalty, horizon,
                discount, domain_name="sysadmin_mdp"):
    nf_name = f"nf_{instance_name}"

    lines = []
    lines.append(f"non-fluents {nf_name} {{")
    lines.append(f"\tdomain = {domain_name};")
    lines.append("\tobjects {")
    lines.append(f"\t\tcomputer : {{{', '.join(computers)}}};")
    lines.append("\t};")
    lines.append("\tnon-fluents {")
    lines.append(f"\t\tREBOOT-PROB = {prob};")
    lines.append(f"\t\tREBOOT-PENALTY = {penalty};")
    for c in computers:
        for y in sorted(incoming[c]):
            lines.append(f"\t\tCONNECTED({y}, {c});")
    lines.append("\t};")
    lines.append("}")
    lines.append("")
    lines.append(f"instance {instance_name} {{")
    lines.append(f"\tdomain = {domain_name};")
    lines.append(f"\tnon-fluents = {nf_name};")
    lines.append("\tinit-state {")
    for c in computers:
        lines.append(f"\t\trunning({c});")
    lines.append("\t};")
    lines.append(f"\tmax-nondef-actions = {len(computers)};")
    lines.append(f"\thorizon = {horizon};")
    lines.append(f"\tdiscount = {discount};")
    lines.append("}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# PPDDL domain + problem
# --------------------------------------------------------------------------

def _fmt(p):
    return f"{p:.6f}"


def _leaf_transition(target, count, total_k, reboot_prob=None):
    if reboot_prob is None:
        p = 0.45 + 0.5 * (1 + count) / (1 + total_k)
    else:
        p = reboot_prob
    p = min(max(p, 0.0), 1.0)
    q = 1.0 - p
    return (f"(probabilistic {_fmt(p)} (running {target}) "
            f"{_fmt(q)} (not (running {target})))")


def _flat_running_case(target, neighbors):
    """Flat (non-nested) enumeration over neighbour running/not-running
    combinations, reproducing RDDL's Bernoulli(.45 + .5*(1+#running)/(1+k))
    for `target`, given `target` is currently running and not rebooted."""
    total_k = len(neighbors)
    if total_k == 0:
        leaf = _leaf_transition(target, 0, 0)
        return f"(when (running {target}) {leaf})"
    clauses = []
    for assignment in itertools.product([True, False], repeat=total_k):
        count = sum(assignment)
        leaf = _leaf_transition(target, count, total_k)
        lits = [f"(running {target})"]
        for y, val in zip(neighbors, assignment):
            lits.append(f"(running {y})" if val else f"(not (running {y}))")
        gd = "(and " + " ".join(lits) + ")"
        clauses.append(f"(when {gd} {leaf})")
    return "(and " + " ".join(clauses) + ")"


def _other_machine_effect(z, neighbors, reboot_prob):
    """Effect fragment for a computer z that is NOT being rebooted this
    epoch: depends on whether z is currently running. Flat `when`s only."""
    running_case = _flat_running_case(z, neighbors)
    not_running_case = _leaf_transition(z, 0, 0, reboot_prob=reboot_prob)
    return (f"(and {running_case} "
            f"(when (not (running {z})) {not_running_case}))")


def _reward_bonus_effect(computers):
    parts = [f"(when (running {c}) (increase (reward) 1.0))" for c in computers]
    return "(and " + " ".join(parts) + ")"


def build_ppddl(instance_name, computers, incoming, prob, penalty, horizon,
                 discount):
    """Time is baked into the action schema (one action per (computer,
    timestep) pair) instead of a shared step-counter fluent that every
    action conditionally increments via `when`. This avoids stacking
    `horizon` extra `when` clauses onto every single action -- those
    were getting folded into the same combinatorial when-compilation
    that the neighbor-transition `when`s already trigger during
    relaxation/heuristic preprocessing, multiplying the blow-up instead
    of just adding to it."""
    domain_name = f"{instance_name}_domain"
    problem_name = instance_name

    step_objs = [f"step{i}" for i in range(horizon + 1)]

    # ---- domain ----
    dlines = []
    dlines.append(f";; Auto-generated PPDDL domain for SysAdmin instance '{instance_name}'.")
    dlines.append(";; Instance-specific: action effects encode this instance's")
    dlines.append(";; network topology. Time is baked into the action schema")
    dlines.append(";; (one action per computer per timestep) rather than tracked")
    dlines.append(";; via a conditionally-incremented fluent, to avoid blowing up")
    dlines.append(";; the relaxation heuristic's when-clause compilation.")
    dlines.append(f"(define (domain {domain_name})")
    dlines.append("  (:requirements :typing :conditional-effects "
                  ":probabilistic-effects :rewards :negative-preconditions)")
    dlines.append("  (:types computer step)")
    dlines.append("  (:constants "
                  f"{' '.join(computers)} - computer "
                  f"{' '.join(step_objs)} - step)")
    dlines.append("  (:predicates (running ?c - computer) (at-step ?s - step))")
    dlines.append("  (:functions (reward))")
    dlines.append("")

    for t in range(horizon):
        cur_step = f"step{t}"
        next_step = f"step{t + 1}"
        for x in computers:
            others = [c for c in computers if c != x]
            effect_parts = [
                f"(running {x})",
                f"(decrease (reward) {_fmt(penalty)})",
                _reward_bonus_effect(computers),
                f"(not (at-step {cur_step}))",
                f"(at-step {next_step})",
            ]
            for z in others:
                effect_parts.append(_other_machine_effect(z, sorted(incoming[z]), prob))
            dlines.append(f"  (:action reboot-{x}-t{t}")
            dlines.append("     :parameters ()")
            dlines.append(f"     :precondition (at-step {cur_step})")
            dlines.append("     :effect (and")
            for part in effect_parts:
                dlines.append(f"        {part}")
            dlines.append("     )")
            dlines.append("  )")
            dlines.append("")

        noop_parts = [
            _reward_bonus_effect(computers),
            f"(not (at-step {cur_step}))",
            f"(at-step {next_step})",
        ]
        for z in computers:
            noop_parts.append(_other_machine_effect(z, sorted(incoming[z]), prob))
        dlines.append(f"  (:action noop-t{t}")
        dlines.append("     :parameters ()")
        dlines.append(f"     :precondition (at-step {cur_step})")
        dlines.append("     :effect (and")
        for part in noop_parts:
            dlines.append(f"        {part}")
        dlines.append("     )")
        dlines.append("  )")
        dlines.append("")

    dlines.append(")")

    ppddl_domain = "\n".join(dlines) + "\n"

    # ---- problem ----
    plines = []
    plines.append(f";; Auto-generated PPDDL problem for SysAdmin instance '{instance_name}'.")
    plines.append(f";; horizon={horizon} is encoded as a reachable goal using a")
    plines.append(";; boolean step marker (step0 .. stepH); each action is tied")
    plines.append(";; to a specific timestep (see domain file).")
    plines.append(";; discount is not representable in PPDDL/SSP form and is not")
    plines.append(f";; applied here (discount={discount} in the RDDL version).")
    plines.append(f"(define (problem {problem_name})")
    plines.append(f"  (:domain {domain_name})")
    plines.append("  (:init")
    for c in computers:
        plines.append(f"     (running {c})")
    plines.append("     (= (reward) 0)")
    plines.append("     (at-step step0)")
    plines.append("  )")
    plines.append(f"  (:goal (at-step step{horizon}))")
    plines.append("  (:metric maximize (reward))")
    plines.append(")")

    ppddl_problem = "\n".join(plines) + "\n"

    return ppddl_domain, ppddl_problem


# --------------------------------------------------------------------------
# Instance generation + CLI
# --------------------------------------------------------------------------
PARAMS = "num_computers num_neighbors reboot_prob horizon"

def build_instances(instance_name, n, k, p, horizon):
    computers, incoming = generate_topology(n, k)
    penalty = 0.75
    discount = 1.0
    rddl = build_rddl(instance_name, computers, incoming, p, penalty, horizon, discount)
    ppddl_domain, ppddl_instance = build_ppddl(instance_name, computers, incoming, p, penalty, horizon, discount)
    return rddl, ppddl_domain, ppddl_instance

def validate_args(dataset, args):
    return True

def create_instances(rddl_dir, instance_name, *args):
    rddl, ppddl_domain, ppddl_problem = build_instances(instance_name, *args)
    # File names
    rddl_file = os.path.join(rddk_dir, instance_name + ".rddl")
    ppddl_dir = rddl_dir.replace("rddl", "ppddl")
    ppddl_file = os.path.join(ppddl_dir, instance_name + ".ppddl")
    # Write RDDL
    os.makedirs(rddl_dir, exist_ok=True)
    with open(rddl_file, "w") as f:
        f.write(rddl)
    print("Created file: " + rddl_file)
    # Write PPDDL
    os.makedirs(ppddl_dir, exist_ok=True)
    with open(ppddl_file, "w") as f:
        f.write(ppddl_domain + "\n")
        f.write(ppddl_problem)
    print("Created file: " + ppddl_file)

if __name__ == "__main__":
    n_args = PARAMS.split() + 2 # plus out_dir and instance_name
    args = sys.argv[1:] # Remove filename
    if len(args) == n_args + 1:
        seed = args.pop(n_args)
        rng.seed(int(seed))
    if len(args) != n_args:
        print("Wrong number of args. Usage: out_dir instance_name " + params + " [seed]")
        sys.exit(-1)
    create_instances(*args)
