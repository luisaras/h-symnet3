
# Converts a RDDLLiftedModel to a Pyperplan Tak
sys.path.append("../pyperplan/")
from task import Task, Operator
from lm_cut import LmCutHeuristic
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.parser.parser import RDDLParser
from pyRDDLGym.core.parser.preprocessor import RDDLPreprocessor
from pyRDDLGym.core.parser.reader import RDDLReader


def flatten(expr):
	if expr[0] == "^":
		left, right = expr[1]
		return flatten(left) | flatten(right)
	if expr[0] == "|":
		left, right = expr[1]
		return flatten(left) | flatten(right)
	if expr[0] == "~":
		return flatten(expr[1][0])
	if expr.is_fluent():
		return {expr.grounded_name}
	if expr.is_or():
		raise NotImplementedError


def extract_rules(expr, path_conditions=None):
	if path_conditions is None:
		path_conditions = set()
	if expr[0] == 'if':
		cond, then_expr, else_expr = expr[1]
		rules = []
		rules.extend(
			extract_rules(
				then_expr,
				path_conditions | flatten(cond)
			)
		)
		rules.extend(
			extract_rules(
				else_expr,
				path_conditions
			)
		)
		return rules
	elif t == 'boolean':
		return [(path_conditions, expr[1])]
	elif t == 'constant':
	return []


class Determinizer:

	def __init__(self, domain_file, instance_file):
		self.task = self.rddl_to_task(domain_file, instance_file)

	def rddl_to_task(self, domain_file, instance_file):
		reader = RDDLReader(domain_file+".rddl", instance_file+".rddl")
		parser = RDDLParser(lexer=None, verbose=False)
		parser.build()
		rddl = parser.parse(reader.rddltxt)
		lifted_model = RDDLLiftedModel(rddl)
		model = RDDLGrounder(lifted_model._AST).ground()

		# Extract rules per variable
		all_rules = {}
		for var_name, (_, expr) in model.cpfs.items():
			all_rules[var_name] = extract_rules(expr)

		# Extract rules per action
		for target_fact, rules in all_rules.items():
			for rule in rules:
				actions = {
					x for x in rule[0] if x in action_fluents
				}
				preconditions = (
					rule[0] - actions
				)
				for action in actions:
					op = operators.setdefault(
						action,
						Operator(...)
					)
					if rule[1]:
						op.add_effects.add(target_fact)
					else:
						op.del_effects.add(target_fact)
					op.preconditions |= precondition
		operators = [Operator(*op) for op in operators.items()]
		return Task("Deterministic Task", facts, initial_state, goals, operators)