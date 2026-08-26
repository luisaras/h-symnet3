#!/usr/bin/env python3
"""
Instance generator for the "Extreme Academic Advising" planning problem.

Courses are organized into `num_levels` levels, with `num_courses` courses per
level, arranged as a DAG: a course at level L (L > 1) has between 1 and
`num_prereqs` prerequisite courses, drawn from courses at *any* earlier level
(1..L-1). Prerequisites (and, separately, the courses that count as "program
requirements") are sampled with probability proportional to the square of the
candidate course's level, so higher (more recent) levels are preferentially
chosen -- matching "courses of higher level are more likely to be a prereq
for courses of the next level". Level-1 courses never have prerequisites.

The "program" itself also has between 1 and `num_prereqs` requirements,
sampled the same weighted way but from the full pool of courses (across all
levels) -- these become the PROGRAM_REQUIREMENT courses that must all be
passed to avoid the PROGRAM_INCOMPLETE_PENALTY.

To match the "Extreme" variant described (reward comes *only* from the
per-time-step penalty for an incomplete program, not from the cost of taking
courses), COURSE_COST and COURSE_RETAKE_COST are set to 0 for every course in
the generated instance, so the only pressure on the agent is to finish the
program's required courses as quickly as possible.

Two equivalent outputs are produced:
  * an RDDL non-fluents/instance block for the `academic_advising_mdp` domain
  * a PPDDL domain + problem pair encoding the same dynamics

PPDDL notes / simplifications (PPDDL has no native support for RDDL's
concurrent actions or for a probability that is a continuous function of
"how many prerequisites have I passed"):
  * Only one course may be taken per PPDDL step (there is also a `wait`
    action that takes no course). The RDDL version allows several courses to
    be attempted in the same semester.
  * For a course with k prerequisites, the pass probability depends only on
    *which* prerequisites are already passed. Since k is small, the action's
    effect is written out as 2^k mutually exclusive `when` branches (one per
    prerequisite-satisfaction pattern), each with the exact probability
        0.05 + 0.95 * (num_satisfied / (1 + k))
    computed to mirror the RDDL cpf for passed'(?c). This grows with 2^k, so
    keep `num_prereqs` modest.
  * Reward is tracked with a numeric fluent `(reward)`; every action (course
    or wait) applies the -5 program-incomplete penalty when appropriate, one
    per simulated time step, mirroring the RDDL reward function.
"""

import sys
import os
import random


PROGRAM_INCOMPLETE_PENALTY = 5.0  # matches the RDDL domain's default magnitude


def course_name(level, idx):
    return f"c{idx}l{level}"


def weighted_sample_without_replacement(candidates, weights, k):
    """Sample k items from candidates without replacement, with selection
    probability at each step proportional to the item's weight."""
    k = min(k, len(candidates))
    pool = list(zip(candidates, weights))
    chosen = []
    for _ in range(k):
        total = sum(w for _, w in pool)
        r = random.uniform(0, total)
        upto = 0.0
        for i, (c, w) in enumerate(pool):
            upto += w
            if upto >= r:
                chosen.append(c)
                pool.pop(i)
                break
        else:
            # Numerical edge case: take the last one.
            chosen.append(pool.pop()[0])
    return chosen


def build_courses_and_prereqs(num_levels, num_courses, num_prereqs):
    courses_by_level = {}
    level_of = {}
    all_courses = []

    for lvl in range(1, num_levels + 1):
        names = [course_name(lvl, idx) for idx in range(1, num_courses + 1)]
        courses_by_level[lvl] = names
        for n in names:
            level_of[n] = lvl
        all_courses.extend(names)

    prereqs = {c: [] for c in all_courses}

    for lvl in range(2, num_levels + 1):
        candidates = [c for l in range(1, lvl) for c in courses_by_level[l]]
        weights = [level_of[c] ** 2 for c in candidates]
        for c in courses_by_level[lvl]:
            k = random.randint(1, max(1, num_prereqs))
            prereqs[c] = weighted_sample_without_replacement(candidates, weights, k)

    # Program requirements: sampled the same weighted way, from all courses.
    weights_all = [level_of[c] ** 2 for c in all_courses]
    k = random.randint(1, max(1, num_prereqs))
    program_reqs = weighted_sample_without_replacement(all_courses, weights_all, k)

    return all_courses, prereqs, program_reqs


def build_rddl(instance_name, courses, prereqs, program_reqs, horizon):
    nf_name = f"nf_{instance_name}"
    lines = []
    lines.append(f"non-fluents {nf_name} {{")
    lines.append("    domain = academic_advising_mdp;")
    lines.append("    objects {")
    lines.append(f"        course : {{{', '.join(courses)}}};")
    lines.append("    };")
    lines.append("    non-fluents {")
    for c in courses:
        for p in prereqs[c]:
            lines.append(f"        PREREQ({p},{c});")
    for r in program_reqs:
        lines.append(f"        PROGRAM_REQUIREMENT({r});")
    # Extreme variant: no cost for taking/retaking courses, only the
    # per-time-step penalty for an incomplete program should drive reward.
    for c in courses:
        lines.append(f"        COURSE_COST({c}) = 0.0;")
        lines.append(f"        COURSE_RETAKE_COST({c}) = 0.0;")
    lines.append(f"        PROGRAM_INCOMPLETE_PENALTY = -{PROGRAM_INCOMPLETE_PENALTY};")
    lines.append("    };")
    lines.append("}")
    lines.append("")
    lines.append(f"instance {instance_name} {{")
    lines.append("    domain = academic_advising_mdp;")
    lines.append(f"    non-fluents = {nf_name};")
    #lines.append(f"    max-nondef-actions = {len(courses)};")
    lines.append(f"    max-nondef-actions = 1;")
    lines.append(f"    horizon = {horizon};")
    lines.append("    discount = 1.0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _take_action(course):
    return f"take-{course}"


def build_ppddl(instance_name, courses, prereqs, program_reqs, horizon):
    domain_name = "academic_advising_ppddl"

    if program_reqs:
        req_conj = " ".join(f"(passed {r})" for r in program_reqs)
        incomplete_cond = f"(not (and {req_conj}))"
        goal_str = f"(and {req_conj})"
    else:
        # Should not happen: k >= 1 always, but keep a well-formed fallback.
        incomplete_cond = "(not (and))"
        goal_str = "(and)"

    # ---------------- domain ----------------
    # `passed` and `taken` are lifted predicates over the `course` type, just
    # like the RDDL pvariables passed(?c) / taken(?c) are lifted over the
    # `course` object type. Grounding uses the same course names as the RDDL
    # instance (e.g. RDDL passed(c1_1) <-> PPDDL (passed c1_1)), so the two
    # encodings refer to exactly the same fluents.
    d = []
    d.append(f"(define (domain {domain_name})")
    d.append("  (:requirements :strips :typing :negative-preconditions :conditional-effects")
    d.append("                 :probabilistic-effects :rewards :equality)")
    d.append("  (:types course)")
    d.append("  (:predicates")
    d.append("    (passed ?c - course)")
    d.append("    (taken ?c - course)")
    d.append("  )")
    d.append("  (:functions (reward))")
    d.append("")

    # No-op / "take nothing this semester" action.
    d.append("  (:action wait")
    d.append("    :parameters ()")
    d.append(f"    :effect (when {incomplete_cond} (decrease (reward) {PROGRAM_INCOMPLETE_PENALTY}))")
    d.append("  )")
    d.append("")

    # For a course with k prerequisites, the pass probability depends on
    # exactly which prerequisites are already passed. Rather than encoding
    # that as a single action with `when` clauses wrapping `probabilistic`
    # effects (legal per the PPDDL grammar, but a construct some heuristic
    # implementations -- ssipp included -- don't handle robustly), each
    # (course, prerequisite-satisfaction-pattern) pair gets its own separate
    # ground action. Every action's :effect is then a flat (and ...) of
    # plain atoms, a single top-level `probabilistic`, and a `when` used
    # only to conditionally decrement the reward -- never `when` wrapping
    # `probabilistic`, and never `probabilistic` wrapping `when`.
    for c in courses:
        plist = prereqs[c]
        k = len(plist)

        if k == 0:
            d.append(f"  (:action {_take_action(c)}")
            d.append("    :parameters ()")
            d.append(f"    :precondition (not (passed {c}))")
            eff = [
                f"(taken {c})",
                f"(probabilistic 0.950000 (passed {c}) 0.050000 (and))",
                f"(when {incomplete_cond} (decrease (reward) {PROGRAM_INCOMPLETE_PENALTY}))",
            ]
            effect_str = "(and\n      " + "\n      ".join(eff) + "\n    )"
            d.append(f"    :effect {effect_str}")
            d.append("  )")
            d.append("")
        else:
            for mask in range(2 ** k):
                sat = [plist[i] for i in range(k) if (mask >> i) & 1]
                unsat = [plist[i] for i in range(k) if not ((mask >> i) & 1)]
                pattern_parts = [f"(passed {p})" for p in sat] + \
                                [f"(not (passed {p}))" for p in unsat]
                prob = round(0.05 + 0.95 * (len(sat) / (1 + k)), 6)
                comp = round(1.0 - prob, 6)

                action_name = f"{_take_action(c)}__p{mask}"
                d.append(f"  (:action {action_name}")
                d.append("    :parameters ()")
                precond_parts = [f"(not (passed {c}))"] + pattern_parts
                d.append("    :precondition (and " + " ".join(precond_parts) + ")")
                eff = [
                    f"(taken {c})",
                    f"(probabilistic {prob:.6f} (passed {c}) {comp:.6f} (and))",
                    f"(when {incomplete_cond} (decrease (reward) {PROGRAM_INCOMPLETE_PENALTY}))",
                ]
                effect_str = "(and\n      " + "\n      ".join(eff) + "\n    )"
                d.append(f"    :effect {effect_str}")
                d.append("  )")
                d.append("")
    d.append(")")
    domain_str = "\n".join(d) + "\n"

    # ---------------- problem ----------------
    p = []
    p.append(f"(define (problem {instance_name})")
    p.append(f"  (:domain {domain_name})")
    p.append("  (:objects")
    p.append("    " + " ".join(courses) + " - course")
    p.append("  )")
    p.append("  (:init")
    p.append("    (= (reward) 0)")
    p.append("  )")
    p.append(f"  (:goal {goal_str})")
    p.append("  (:metric maximize (reward))")
    p.append(")")
    problem_str = "\n".join(p) + "\n"

    return domain_str, problem_str


def generate_instance(instance_name, num_levels, num_courses, num_prereqs, horizon):
    courses, prereqs, program_reqs = build_courses_and_prereqs(
        num_levels, num_courses, num_prereqs
    )
    rddl = build_rddl(instance_name, courses, prereqs, program_reqs, horizon)
    ppddl_domain, ppddl_problem = build_ppddl(
        instance_name, courses, prereqs, program_reqs, horizon
    )
    return rddl, ppddl_domain, ppddl_problem


if __name__ == "__main__":
    args = sys.argv[1:]
    n_args = 6
    if len(args) == n_args + 1:
        seed = args.pop(n_args)
        random.seed(int(seed))
    if len(args) != n_args:
        print("Wrong number of args. Usage: out-dir instance_name num_levels num_courses num_prereqs horizon [seed]")
        sys.exit(-1)
    out_dir = args[0]
    instance_name = args[1]  # Name of both PPDDL and RDDL instances.
    num_levels = int(args[2])  # Number of levels.
    num_courses = int(args[3])  # Number of courses per level.
    num_prereqs = int(args[4])  # Max number of prereqs. The mimimum is 1 (except for first level).
    horizon = int(args[5])
    rddl, ppddl_domain, ppddl_problem = generate_instance(instance_name, num_levels, num_courses, num_prereqs, horizon)
    os.makedirs(out_dir, exist_ok=True)
    rddl_file = os.path.join(out_dir, instance_name + ".rddl")
    out_dir = out_dir.replace("rddl", "ppddl")
    os.makedirs(out_dir, exist_ok=True)
    ppddl_file = os.path.join(out_dir, instance_name + ".ppddl")
    with open(rddl_file, "w") as f:
        f.write(rddl)
    with open(ppddl_file, "w") as f:
        f.write(ppddl_domain + "\n")
        f.write(ppddl_problem)
    print("Generated file: " + rddl_file)
    print("Generated file: " + ppddl_file)