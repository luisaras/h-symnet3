# ASNet code.
import importlib
import os
import re
import subprocess

import ssipp  # noqa: F811

ABOVE_DIR = os.path.abspath(
	os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SSIPP_MANUAL_DIR = os.path.join(ABOVE_DIR, 'ssipp')


def has_ssipp_solver():
	"""Check whether we have solver_ssp somewhere (this is the SSiPP planner
	binary, not the SSiPP Python library)."""
	try:
		get_ssipp_solver_path_auto()
		return True
	except (FileNotFoundError, ImportError):
		return False


def try_install_ssipp_solver():
	"""Try to install SSiPP solver (i.e planner binary, not library) if it is
	not available already."""

	# TODO: make this installer less racy (& do same for try_install_fd)

	if has_ssipp_solver():
		return

	print("Installing SSiPP's solver to %s" % SSIPP_MANUAL_DIR)
	if not os.path.exists(SSIPP_MANUAL_DIR):
		subprocess.run([
			"git", "clone", "https://gitlab.com/qxcv/ssipp.git",
			SSIPP_MANUAL_DIR
		],
					   check=True,
					   cwd=ABOVE_DIR)
		subprocess.run(["python", "build.py", "solver_ssp"],
					   check=True,
					   cwd=SSIPP_MANUAL_DIR)
	ssipp_binary_path = os.path.join(SSIPP_MANUAL_DIR, "solver_ssp")
	assert os.path.exists(ssipp_binary_path), \
		"install failed; nothing found at '%s'" % (ssipp_binary_path, )


def get_ssipp_solver_path_auto():
	"""Automagically get path to SSiPP solver_ssp by assuming it's in the same
	directory as the SSiPP Python module, or that it has been downloaded into
	the current dir. Let current dir take preference."""
	# first check current dir
	current_dir_solver = os.path.join(SSIPP_MANUAL_DIR, 'solver_ssp')
	if os.path.exists(current_dir_solver):
		return current_dir_solver

	# if that failed, we check the other dir
	ssipp_spec = importlib.util.find_spec('ssipp')
	suggestion = "Maybe it's easier to compile SSiPP manually and use " \
		"--ssipp-path to specify path? Or use try_install_ssipp_solver()?"
	if ssipp_spec is None:
		raise ImportError(
			"Could not import SSiPP to do auto-magic solver_ssp path "
			"detection. " + suggestion)
	ssipp_dir = os.path.dirname(ssipp_spec.origin)
	solver_ssp_path = os.path.join(ssipp_dir, 'solver_ssp')
	if not os.path.exists(solver_ssp_path):
		raise FileNotFoundError(
			"Could not auto-magically detect SSiPP solver_ssp at '%s'. %s" %
			(solver_ssp_path, suggestion))

	return solver_ssp_path


class Evaluator:
	# FIXME: the name of this class is terrible. What does it actually do? Is
	# it just evaluating heuristics, or is it planning underneath? Resolve &
	# rename!
	def __init__(self, planner_exts, heuristic_name):
		self._ssipp = planner_exts.ssipp
		self.problem = planner_exts.ssipp_problem
		ssp = self._ssipp.SSPfromPPDDL(self.problem)
		heuristic = self._ssipp.createHeuristic(ssp, heuristic_name)
		self.evaluator = self._ssipp.SuccessorEvaluator(heuristic)

	def eval_state(self, ssipp_state):
		return self.evaluator.state_value(ssipp_state)

	def succ_probs_vals(self, ssipp_state, action_name):
		action = self.problem.find_action("(" + action_name + ")")
		assert action is not None, "could not find %r" % (action_name, )
		return [(e.probability, e.value)
				for e in self.evaluator.succ_iter(ssipp_state, action)]


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

class SSiPPDataGenerator(ActionDataGenerator):
	"""Basic class for generators which use SSiPP"""

	def __init__(self, mod_sandbox):
		# important to have only weak ref to sandbox because the sandbox also
		# has a ref to us (!)
		self.mod_sandbox = weak_ref_to(mod_sandbox)
		self.ssipp_problem = weak_ref_to(self.mod_sandbox.ssipp_problem)

class LMCutDataGenerator(SSiPPDataGenerator):
	"""Adds 'this is in a disjunctive cut'-type flags to propositions."""
	extra_dim = 3
	dim_names = ['in-any-cut', 'in-singleton-cut', 'in-last-cut']
	IN_ANY_CUT = 0
	IN_SINGLETON_CUT = 1
	# The last cut contains actions helpful actions at the current state, and
	# the first cut contains the final goal-achieving action. That's just a
	# convention in my SSiPP wrapper, of course.
	IN_LAST_CUT = 2

	def __init__(self, *args):
		super().__init__(*args)
		self.cutter = Cutter(self.mod_sandbox)

	def get_extra_data_no_memory(self, cstate):
		out_vec = np.zeros((len(cstate.acts_enabled), self.extra_dim))
		ssipp_state = cstate.to_ssipp(self.mod_sandbox)
		cuts = self.cutter.get_action_cuts(ssipp_state)
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


class PlannerExtensions(object):
	"""Wrapper to hold references to SSiPP and MDPSim modules, and references
	to the relevant loaded problems (like the old ModuleSandbox). Mostly
	keeping this because it makes it convenient to pass stuff around, as I
	often need SSiPP and MDPSim at the same time."""

	heur_map = {
	  "lmc": "lm-cut",
	}

	def __init__(self,
				 ppddl_file,
				 instance_name,
				 heuristics):
		# SSiPP stuff
		ssipp.readPDDLFile(ppddl_file)
		self.ssipp_problem = ssipp.init_problem(instance_name)
		# this leaks for some reason; will store it here so I don't have to
		# reconstruct
		self.ssipp_ssp_iface = ssipp.SSPfromPPDDL(self.ssipp_problem)

		self.heuristics = [Evaluator(weak_ref_to(self), heur_map[h]) for h in heuristics]


	@property
	def ssipp_dead_end_value(self):
		return ssipp.get_dead_end_value()

	def compute_heuristics(self, state, instance_parser):
		state = self.convert_state(state, instance_parser)
		return [heur.eval_state(state) for heur in self.heuristics]

	def convert_state(self, state, ip):
		"""Converts true prop list to string format that SSiPP can read.
		all_props can be obtained from an MDPSimObservation instance's props_true
		attributes."""

		format_props = []
        if len(ip.unpara_fluents) != 0:
            for (i, st) in enumerate(sorted(ip.unpara_fluents)):
                if state[ip.state_to_num[st]] == 1: # true?
                	format_props.add(st)

        for st in ip.para_state_names:  # For each fluent
            for node in ip.state_object_names:  # For each parameter of the fluent (a vertex in the graph - rememberd dbn)
                stn = st + '(' + node + ')'
                try:  # Assign features from the mapping of nodes and states to indices
                	val = float(state[ip.state_to_num[stn]])
                    if val == 1:
                    	format_props.add(st + " " + node.replace(",",""))
                except KeyError as e:
                    pass

        return ', '.join(format_props)

		#format_props = []
		#for prop_obj, truth in all_props:
		#	if not truth:
		#		continue
		#	old_prop = prop_obj.identifier
		#	assert old_prop[0] == '(', old_prop
		#	assert old_prop[-1] == ')', old_prop
		#	tokens = old_prop[1:-1].split()
		#	name = tokens[0]
		#	args = tokens[1:]
		#	format_props.append('%s %s' % (name, ' '.join(args)))
		return ', '.join(format_props)
