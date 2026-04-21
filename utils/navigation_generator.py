import random
import os

def cell_name(i, j):
    return f"c_{i}_{j}"

def generate_instance(D, horizon, instance_name, out_dir):
    cells = [(i, j) for i in range(D) for j in range(D)]

    start = random.choice(cells)
    goal = random.choice(cells)

    while goal == start:
        goal = random.choice(cells)

    objects = ", ".join([cell_name(i, j) for i, j in cells])

    nonfluents = []
    init_state = []

    # Goal
    nonfluents.append(f"is_goal({cell_name(*goal)});")

    # Adjacency
    for i, j in cells:
        if i > 0:
            nonfluents.append(f"north({cell_name(i, j)}, {cell_name(i-1, j)});")
        if i < D-1:
            nonfluents.append(f"south({cell_name(i, j)}, {cell_name(i+1, j)});")
        if j > 0:
            nonfluents.append(f"west({cell_name(i, j)}, {cell_name(i, j-1)});")
        if j < D-1:
            nonfluents.append(f"east({cell_name(i, j)}, {cell_name(i, j+1)});")

    # Initial robot position
    init_state.append(f"robot_at({cell_name(*start)});")

    content = f"""
instance {instance_name} {{
    domain = navigation_mdp;

    objects {{
        cell : {{{objects}}};
    }}

    non-fluents {{
        {" ".join(nonfluents)}
    }}

    init-state {{
        {" ".join(init_state)}
    }}

    horizon = {horizon};
    discount = 1.0;
}}
"""

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, instance_name + ".rddl"), "w") as f:
        f.write(content)


def generate_dataset(name, Dmin, Dmax, horizon, n_instances, out_dir):
    for i in range(n_instances):
        D = random.randint(Dmin, Dmax)
        generate_instance(D, horizon, f"{name}_{i}", out_dir)


if __name__ == "__main__":
    random.seed(0)

    generate_dataset("train", 9, 14, 40, 50, "instances/train")
    generate_dataset("val", 15, 18, 60, 20, "instances/val")
    generate_dataset("test", 20, 25, 60, 20, "instances/test")