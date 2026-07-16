#!/bin/python3
"""
Gerador unificado (RDDL + PPDDL) de instancias do dominio Navigation.

Gera, a partir dos MESMOS parametros e da MESMA seed, duas
representacoes com semantica identica:

  * RDDL  : {instance_name}.rddl               (non-fluents + instance,
            usa o dominio compartilhado navigation_mdp.rddl)
  * PPDDL : {instance_name}_domain.pddl
            {instance_name}_problem.pddl

Os valores de P(x,y) (probabilidade de o robo desaparecer ao entrar na
celula) sao sorteados UMA UNICA VEZ (na mesma ordem de chamadas de
random.* do gerador RDDL original) e reaproveitados na construcao das
duas representacoes, garantindo que RDDL e PPDDL descrevam exatamente
o mesmo MDP -- nao apenas com a mesma formula, mas com os mesmos
numeros sorteados.

Semantica (identica a navigation_mdp.rddl):
  1) Se o robo ja esta no GOAL, ele permanece la para sempre
     (absorvente), independente da acao escolhida.
  2) Caso contrario, se a acao aponta para uma direcao valida (existe
     vizinho naquela direcao a partir da posicao atual do robo):
       - a celula de origem e SEMPRE esvaziada (fica false);
       - a celula de destino recebe Bernoulli(1 - P(destino)): com
         probabilidade P(destino) o robo "desaparece" (nao chega a
         lugar nenhum), com probabilidade 1-P(destino) ele chega.
  3) Se a direcao escolhida e invalida (borda do grid), nada muda: o
     robo permanece onde estava.
  4) P(x,y) so e definido (potencialmente > 0) para linhas interiores
     (2..h-1); a linha inicial (y1) e a linha do goal (yh) sao sempre
     seguras (P = 0).
  5) Recompensa = -1 a cada passo em que o robo nao esta no goal, 0
     quando esta.

Nota sobre PPDDL: o PPDDL classico (Younes / mdpsim) exige que as
probabilidades em efeitos `probabilistic` sejam literais numericos, e
nao podem depender de fluentes numericos em tempo de execucao. Como
P(x,y) varia por celula, o dominio PPDDL gerado e, assim como os
non-fluents do RDDL, especifico de cada instancia: cada celula/direcao
vira uma acao grounded com a probabilidade ja embutida.
"""
import os, sys, random


# ---------------------------------------------------------------------
# Sorteio de P(x,y) -- executado uma unica vez e compartilhado entre
# RDDL e PPDDL. Reproduz EXATAMENTE a ordem de chamadas de random.* do
# gerador RDDL original (navigation.py), para que ambos os formatos
# resultem nos mesmos valores dado o mesmo seed.
# ---------------------------------------------------------------------
def compute_P(w, h, t):
	if t == "deterministic":
		# probability = 0
		safe_cols = []
		danger_chance = lambda i: 0
		safe_chance = lambda i: 0
	elif t == "default":
		# rddlsim generator probabilities
		safe_cols = range(1, w + 1)
		danger_chance = lambda i: (0.01 + ((0.9 * (i - 1)) / (w - 1))) + 0.05 * random.uniform(0, 1)
		safe_chance = danger_chance
	else:
		# symnet probabilities
		danger_chance = lambda i: random.uniform(0.88, 0.92)
		safe_chance = lambda i: random.uniform(0.045, 0.055)
		if t == "stochastic":
			# random single safe column
			safe_cols = [random.randint(1, w)]
		else:  # corridor
			# first column as the safe column
			safe_cols = [1]

	P = {}
	for i in range(1, w + 1):
		if i in safe_cols:
			for j in range(2, h):
				p = safe_chance(i)
				if p > 0:
					P[(i, j)] = p
		else:
			for j in range(2, h):
				p = danger_chance(i)
				if p > 0:
					P[(i, j)] = p
	return P


# ---------------------------------------------------------------------
# RDDL
# ---------------------------------------------------------------------
def build_rddl(instance_name, w, h, horizon, P):
	xpos = [f'x{i}' for i in range(1, w + 1)]
	ypos = [f'y{j}' for j in range(1, h + 1)]
	nonfluents = []
	for i in range(1, w + 1):
		if i > 1:
			nonfluents.append(f"WEST(x{i},x{i-1});")
		if i < w:
			nonfluents.append(f"EAST(x{i},x{i+1});")
	for j in range(1, h + 1):
		if j > 1:
			nonfluents.append(f"SOUTH(y{j},y{j-1});")
		if j < h:
			nonfluents.append(f"NORTH(y{j},y{j+1});")
	nonfluents.append("MIN-XPOS(x1);")
	nonfluents.append("MIN-YPOS(y1);")
	nonfluents.append(f"MAX-XPOS(x{w});")
	nonfluents.append(f"MAX-YPOS(y{h});")
	nonfluents.append(f"GOAL(x{w},y{h});")

	for i in range(1, w + 1):
		for j in range(2, h):
			p = P.get((i, j), 0.0)
			if p > 0:
				nonfluents.append(f"P(x{i},y{j}) = {p};")

	xpos_str = ",".join(xpos)
	ypos_str = ",".join(ypos)
	nonfluents_str = "\n\t\t".join(nonfluents)

	rddl = f"""non-fluents nf_{instance_name} {{
	domain = navigation_mdp;
	objects {{
		xpos : {{{xpos_str}}};
		ypos : {{{ypos_str}}};
	}};
	non-fluents {{
		{nonfluents_str}
	}};
}}

instance {instance_name} {{
	domain = navigation_mdp;
	non-fluents = nf_{instance_name};
	init-state {{
		robot-at(x{w},y1);
	}};
	max-nondef-actions = 1;
	horizon = {horizon};
	discount = 1.0;
}}"""
	return rddl


# ---------------------------------------------------------------------
# PPDDL
# ---------------------------------------------------------------------
_DIRS = {
	"north": (0, 1),
	"south": (0, -1),
	"east": (1, 0),
	"west": (-1, 0),
}


def build_ppddl(instance_name, w, h, horizon, P):
	goal = (w, h)
	init_pos = (w, 1)

	xpos = [f"x{i}" for i in range(1, w + 1)]
	ypos = [f"y{j}" for j in range(1, h + 1)]

	def robot_at(i, j):
		return f"(robot-at x{i} y{j})"

	actions = []
	for i in range(1, w + 1):
		for j in range(1, h + 1):
			for dname, (dx, dy) in _DIRS.items():
				aname = f"move-{dname}-x{i}-y{j}"
				precond = robot_at(i, j)

				if (i, j) == goal:
					# Absorvente: qualquer acao no goal e um no-op com
					# recompensa 0, independente da direcao escolhida.
					effect = "(increase (reward) 0)"
					actions.append((aname, precond, effect))
					continue

				i2, j2 = i + dx, j + dy
				valid = 1 <= i2 <= w and 1 <= j2 <= h

				if not valid:
					# Bate na parede: nada muda, custo -1 (nao esta no goal).
					effect = "(decrease (reward) 1)"
					actions.append((aname, precond, effect))
					continue

				p = P.get((i2, j2), 0.0)
				dest_is_goal = (i2, j2) == goal
				reward_success = 0 if dest_is_goal else 1

				vacate = f"(not {robot_at(i, j)})"
				arrive = robot_at(i2, j2)

				if p > 0.0:
					effect = (
						f"(and {vacate}\n"
						f"\t\t\t(probabilistic\n"
						f"\t\t\t\t{1.0 - p} (and {arrive} (decrease (reward) {reward_success}))\n"
						f"\t\t\t\t{p} (decrease (reward) 1)))"
					)
				else:
					effect = f"(and {vacate} {arrive} (decrease (reward) {reward_success}))"

				actions.append((aname, precond, effect))

	action_blocks = []
	for aname, precond, effect in actions:
		action_blocks.append(
			f"\t(:action {aname}\n"
			f"\t\t:parameters ()\n"
			f"\t\t:precondition {precond}\n"
			f"\t\t:effect {effect}\n"
			f"\t)"
		)
	actions_str = "\n\n".join(action_blocks)

	domain = f"""(define (domain {instance_name}_domain)
	(:requirements :typing :probabilistic-effects :rewards :conditional-effects)

	(:types xpos ypos)

	(:predicates
		(robot-at ?x - xpos ?y - ypos)
	)

	(:functions
		(reward)
	)

{actions_str}
)
"""

	objects_str = "\n\t\t" + " ".join(xpos) + " - xpos" + \
		"\n\t\t" + " ".join(ypos) + " - ypos"

	problem = f"""(define (problem {instance_name})
	(:domain {instance_name}_domain)

	(:objects{objects_str}
	)

	(:init
		{robot_at(*init_pos)}
		(= (reward) 0)
	)

	(:goal {robot_at(*goal)})
	(:goal-reward 0)
	(:metric maximize (reward))

	;; extensoes usadas por simuladores IPPC-style (mdpsim/prost),
	;; ignoradas por parsers estritamente PPDDL1.0
	(:horizon {horizon})
	(:discount-factor 1.0)
)
"""
	return domain, problem


# ---------------------------------------------------------------------
# Ponto de entrada compartilhado
# ---------------------------------------------------------------------
def generate_instance(instance_name, w, h, t, horizon):
	"""Sorteia P uma unica vez e gera as tres representacoes a partir
	dele: RDDL, dominio PPDDL e problema PPDDL."""
	P = compute_P(w, h, t)
	rddl = build_rddl(instance_name, w, h, horizon, P)
	ppddl_domain, ppddl_problem = build_ppddl(instance_name, w, h, horizon, P)
	return rddl, ppddl_domain, ppddl_problem


if __name__ == "__main__":
	args = sys.argv[1:]
	if len(args) == 7:
		seed = args.pop(6)
		random.seed(int(seed))
	if len(args) != 6:
		print("Wrong number of args. Usage: out-dir instance_name width height type horizon [seed]")
		sys.exit(-1)

	out_dir = args[0]
	instance_name = args[1]
	width = int(args[2])
	height = int(args[3])
	type = args[4]
	horizon = int(args[5])

	rddl, ppddl_domain, ppddl_problem = generate_instance(instance_name, width, height, type, horizon)

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
