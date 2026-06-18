#!/bin/python3
import os, sys, random

def generate_instance(instance_name, w, h, t, horizon):
	xpos = [f'x{i}' for i in range(1,w+1)]
	ypos = [f'y{i}' for i in range(1,h+1)]
	nonfluents = []
	for i in range(1,w+1):
		if i > 1:
			nonfluents.append(f"WEST(x{i},y{i-1});")
		if i < w:
			nonfluents.append(f"EAST(x{i},x{i+1});")
	for j in range(1,h+1):
		if j > 1:
			nonfluents.append(f"SOUTH(y{j},y{j-1});")
		if j < h:
			nonfluents.append(f"NORTH(y{j},y{j+1});")
	nonfluents.append("MIN-XPOS(x1);")
	nonfluents.append("MIN-YPOS(y1);")
	nonfluents.append(f"MAX-XPOS(x{w});")
	nonfluents.append(f"MAX-YPOS(y{h});")
	nonfluents.append(f"GOAL(x{w},y{h});")

	# P
	if t == "deterministic":
		safe_cols = []
		danger_chance = lambda i : 0
	elif t == "default":
		safe_cols = range(1, w+1)
		danger_chance = lambda i : (0.01 + ((0.9*(i-1))/(w - 1))) + 0.05*random.uniform(0,1)
		safe_chance = danger
	else:
		danger_chance = lambda i : random.uniform(0.88, 0.92)
		safe_chance = lambda i : random.uniform(0.045, 0.055)
		if t == "stochastic":
			safe_cols = [random.randint(1, w)]
		else: # corridor
			safe_cols = [1]
	for i in range(1, w+1):
		if i in safe_cols:
			for j in range(2, h):
				p = safe_chance(i)
				if p > 0:
					nonfluents.append(f"P(x{i}, y{j}) = {p};")
		else:
			for j in range(2, h):
				p = danger_chance(i)
				if p > 0:
					nonfluents.append(f"P(x{i}, y{j}) = {p};")

	xpos = ",".join(xpos)
	ypos = ",".join(ypos)
	nonfluents = "\n\t\t".join(nonfluents)

	return f"""
non-fluents nf_{instance_name} {{
	domain = navigation_mdp;
	objects {{
		xpos : {{{xpos}}};
		ypos : {{{ypos}}};
	}};
	non-fluents {{
		{nonfluents}
	}};
}}

instance {instance_name} {{
	domain = navigation_mdp;
	non-fluents = nf_{instance_name};
	init-state {{
		robot-at(x{w-1},y1);
	}};
	max-nondef-actions = 1;
	horizon = {horizon};
	discount = 1.0;
}}"""


if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) == 7:
		seed = args.pop(6)
		random.seed(int(seed))
	if len(args) != 6:
		print("Wrong number of args")
		sys.exit(-1)
	out_dir = args.pop(0)
	instance_name = args[0]
	x = int(args[1])
	y = int(args[2])
	t = args[3]
	h = int(args[4])
	content = generate_instance(instance_name, x, y, t, h)
	os.makedirs(out_dir, exist_ok=True)
	with open(os.path.join(out_dir, instance_name + ".rddl"), "w") as f:
		f.write(content)