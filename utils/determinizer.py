
# Converts a RDDLLiftedModel to a Pyperplan Tak
sys.path.append("../pyperplan/")
from task import Task, Operator
from pyRDDLGym.core.grounder import RDDLGrounder


def flatten(expr):
    if expr.is_and():
        return flatten(expr.left) | flatten(expr.right)

    if expr.is_exists():
        return flatten(expr.body)

    if expr.is_fluent():
        return {expr.grounded_name}

    if expr.is_or():
        raise NotImplementedError


def extract_rules(expr, path_conditions=None):
    if path_conditions is None:
        path_conditions = set()

    t, val = expr.etype()
    if t == 'control':

        cond, then_expr, else_expr = expr.args

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

    elif t == 'constant':
        return [(path_conditions, self._expr[1])]

    elif t == 'constant':

    return []


class Determinizer:

	def __init__(self, lifted_model):
		self.lifted_model = model

	def to_task(self):
		model = RDDLGrounder(self.lifted_model._AST).ground()

		all_rules = {}
		for var_name, (_, expr) in model.cpfs.items():
			all_rules[var_name] = extract_rules(expr)

		for target_fact, rules in all_rules.items():

		    for rule in rules:

		        actions = {
		            x for x in rule[0]
		            if x in action_fluents
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