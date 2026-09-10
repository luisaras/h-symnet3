# ASNet code.
import importlib
import sys, os, re, subprocess
from weakref import proxy, ProxyTypes

def weak_ref_to(obj):
	"""Create a weak reference to object if object is not a weak reference. If
	object is a weak reference, then return that reference unchanged."""
	if obj is None or isinstance(obj, ProxyTypes):
		return obj
	return proxy(obj)


def num_heuristic_features(heuristic_names):
	return len(heuristic_names)

class PlannerExtensions(object):
	"""Wrapper to hold references to SSiPP and MDPSim modules, and references
	to the relevant loaded problems (like the old ModuleSandbox). Mostly
	keeping this because it makes it convenient to pass stuff around, as I
	often need SSiPP and MDPSim at the same time."""

	heur_map = {
	  "lmc": "lm-cut",
	  "lmc3": "lm-cut3",
	}

	def __init__(self,
				 ppddl_files, # instance + domain
				 instance_name,
				 heuristics):
		import ssipp
		# SSiPP stuff
		print(f"Initializing {instance_name} PPDDL problem...", flush=True)
		for file in ppddl_files:
			ssipp.readPDDLFile(file)
		self.ssipp = ssipp
		self.ssipp_problem = ssipp.init_problem(instance_name)
		if self.ssipp_problem == None:
			print("Error while initializing the instance: " + instance_name)
			sys.exit(1)
		# this leaks for some reason; will store it here so I don't have to reconstruct
		self.ssp = ssipp.SSPfromPPDDL(self.ssipp_problem)
		print(f"PPDDL {instance_name} initialized.", flush=True)

		self.heuristics = [Evaluator(weak_ref_to(self), PlannerExtensions.heur_map[h]) for h in heuristics]
		self._cache = dict()

	def compute_heuristics(self, state):
		if state in self._cache:
			return self._cache[state]
		else:
			val = [heur.eval_state(state) for heur in self.heuristics]
			self._cache[state] = val
			return val


class Evaluator:
	# FIXME: the name of this class is terrible. What does it actually do? Is
	# it just evaluating heuristics, or is it planning underneath? Resolve &
	# rename!
	def __init__(self, planner_exts, heuristic_name):
		print(f"Initializing heuristic evaluator {heuristic_name}... ", flush=True)
		self.ssipp_problem = planner_exts.ssipp_problem
		self.heuristic = planner_exts.ssipp.createHeuristic(planner_exts.ssp, heuristic_name)
		#self.evaluator = planner_exts.ssipp.SuccessorEvaluator(self.heuristic)
		#self.cutter = Cutter(planner_exts)
		print(heuristic_name + " initialized.", flush=True)

	def eval_state(self, ssipp_state):
		ssipp_state = self.ssipp_problem.get_intermediate_state(ssipp_state)
		#cuts = self.cutter.get_action_cuts(ssipp_state)
		#print("=========== CUTS: " + str(cuts))
		#return self.evaluator.state_value(ssipp_state)
		return [self.heuristic.value(ssipp_state)]

class Cutter:
	# ssipp appends -prob-j to an action name to signify that it is the j-th
	# (determinised) outcome of the original action. -prob-j-prec-i is used for
	# the j-th determinised outcome of the i-th (disjunctive) case.

	act_re = re.compile(r'^(\(.+?\))(?:-(?:prob|prec|c)-\d+)*$')

	def __init__(self, planner_exts):
		self.problem = planner_exts.ssipp_problem
		self.lm_cut = planner_exts.ssipp.LMCutHeuristic(self.problem)
		# we cache cuts forever
		self.cut_cache = {}

	def real_action_name(self, act_name):
		match = self.act_re.match(act_name)
		if match is None:
			raise ValueError("Couldn't parse action name '%s'" % act_name)
		group, = match.groups()
		return group

	def get_action_cuts(self, ssipp_state):
		if ssipp_state not in self.cut_cache:
			cuts_value = self.lm_cut.valueAndCuts(ssipp_state)
			new_cuts = []
			for cut in cuts_value.cuts:
				new_cut = frozenset(
					self.real_action_name(name) for name in cut)
				new_cuts.append(new_cut)
			self.cut_cache[ssipp_state] = new_cuts
		return self.cut_cache[ssipp_state]


def strip_parens(thing):
    """Convert string of form `(foo bar baz)` to `foo bar baz` (i.e. strip
    leading & trailing parens). More complicated than it should be b/c it does
    safety checks to catch my bugs :)"""
    assert len(thing) > 2 and thing[0] == "(" and thing[-1] == ")", \
        "'%s' does not look like it's surrounded by parens" % (thing,)
    stripped = thing[1:-1]
    assert "(" not in stripped and ")" not in stripped, \
        "parens in '%s' aren't limited to start and end" % (thing,)
    return stripped


class LMCutDataGenerator:
	"""Adds 'this is in a disjunctive cut'-type flags to propositions."""
	extra_dim = 3
	dim_names = ['in-any-cut', 'in-singleton-cut', 'in-last-cut']
	IN_ANY_CUT = 0
	IN_SINGLETON_CUT = 1
	# The last cut contains actions helpful actions at the current state, and
	# the first cut contains the final goal-achieving action. That's just a
	# convention in my SSiPP wrapper, of course.
	IN_LAST_CUT = 2

	def __init__(self, planner_exts):
		self.planner_exts = planner_exts
		self.cutter = Cutter(planner_exts)

	def get_extra_data_no_memory(self, cstate):
		out_vec = np.zeros((len(cstate.acts_enabled), self.extra_dim))
		ssipp_state = cstate.to_ssipp(self.planner_exts)
		cuts = self.cutter.get_action_cuts(ssipp_state)

	def get_cuts_heur(cuts):
		in_unary_cut = set()
		in_any_cut = set()
		for cut in cuts:
			cut = frozenset(strip_parens(a) for a in cut)
			if len(cut) == 1:
				in_unary_cut.update(cut)
			if len(cut) >= 1:
				# all actions in cuts (unary cuts or not) end up here
				in_any_cut.update(cut)
		if cuts:
			in_last_cut = {strip_parens(a) for a in cuts[-1]}
		else:
			in_last_cut = set()
		all_act_names = [a.unique_ident for a, _ in cstate.acts_enabled]
		assert (in_unary_cut | in_any_cut) <= set(all_act_names), \
			"there are some things in cuts that aren't in action set"
		assert in_last_cut <= set(all_act_names), \
			"there are things in the last cut that aren't really actions (?!)"
		for idx, act_name in enumerate(all_act_names):
			if act_name in in_unary_cut:
				out_vec[idx][self.IN_SINGLETON_CUT] = 1
			if act_name in in_any_cut:
				out_vec[idx][self.IN_ANY_CUT] = 1
			if act_name in in_last_cut:
				out_vec[idx][self.IN_LAST_CUT] = 1
		return out_vec
		

class ActionCountDataGenerator:
    """Counts number of times each action has been executed so far."""
    extra_dim = 1
    dim_name = 'action_count'
    dim_names = [dim_name]
    requires_memory = True

    def __init__(self, problem_meta):
        self.problem_meta = weak_ref_to(problem_meta)

    def get_extra_data_with_memory(self, this_cstate, prev_cstate, prev_act,
                                   is_init_cstate):
        if is_init_cstate:
            return np.zeros((len(this_cstate.acts_enabled), self.extra_dim))
        # FIXME: this is a really dumb way to do things; I should be keeping
        # track of this separately & passing it into ALL memory-based
        # functions. Also I should keep aux_data as 2D instead of 1D, since
        # that's less error-prone. Fix this if I ever need to do it again.
        extra_dim = max(prev_cstate._aux_data_interp_to_id.values()) + 1
        old_aux_reshaped = prev_cstate.aux_data.reshape((-1, extra_dim))
        prev_dim_id = prev_cstate._aux_data_interp_to_id[self.dim_name]
        # get previous count vector & increment relevant count by 1
        aux_data_1d = old_aux_reshaped[:, prev_dim_id].copy()
        act_id = self.problem_meta.act_unique_id_to_index(
            prev_act.unique_ident)
        aux_data_1d[act_id] += 1
        aux_data_2d = aux_data_1d.reshape((-1, 1))
        return aux_data_2d
