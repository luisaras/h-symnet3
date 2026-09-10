# =============================================================================
# General settings for flags
# =============================================================================

net_config = None # For a custom symnet3_config.py
restore_config = False # Save/load config when reloading trained net

# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------

'''
IPPC domains:
academic_advising_ippc(Acad), navigation(Nav), game_of_life(GoL), crossing_traffic(CT), wildfire(Wild), triangle_tireworld(TT), 
elevators(Elev), sysadmin(Sys), traffic(Traf), recon(Recon), skill_teaching(Skill), tamarisk(Tam)

LR domains:
academic_advising(EAcad), navigation(DNav), corridor(StNav), recon(SRecon), pizza_delivery_windy(Pizza), stochastic_wall(StNav)
'''

setting = "custom" # "ippc" or "lr" to select specific instances
domain = "navigation" # Domain to be tested on
benchmark_folder = "../../benchmarks/"

num_validation_episodes = 5 # Validation episodes for each epoch
num_testing_episodes = 15 # Testing episodes when model has been trained

trajectory_dataset_folder = "../../data/datasets/"
heuristics_dataset_folder = "../../data/heuristics/"
max_transitions_per_instance = 300
last_in_dataset = False # Setting to true might lead to better results since PROST learns while executing

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

model_dir =  "models/" # Path to model
exp_description = "standard" # Suffix for model folder

# Train settings
batch_size = 32
lr = 0.001
grad_clip_value = 5.0
ckpt_freq = 10 # After how many epoch is an evaluation performed
keep_ckpts = True # Keeps all checkpoints instead of only the best-performing one

# Best not to change these. 
add_separate_adj = False # Keep DBN edges separately in two extra adjacency layers
add_edge_type = True # An extra adjacency layer for each edge type
remove_dbn = False # Discard DBN layers
merged_model = True # Add new edges between new nodes for non-fluents and gnd objects
split_dbn = False # Add layer of type edges (one layer for each type pair of connected nodes)
use_type_encoding = True # Add a features indicating the type of a node
use_fluent_for_kl = True # Only use random fluent nodes when computing KL
repeat_graph_nf = True # Add graph nonfluents as node features
num_threads = 4 # Number of threads to parallelize testing
add_aux_loss = False # Use KL (False for default SymNet3.0 and SymNet2.0)
decay_aux_loss = False # Use KL decay

# -----------------------------------------------------------------------------
# Planning Heuristics
# -----------------------------------------------------------------------------

# Planning heuristics
heuristics = [] # "lmc", "hadd", and/or "hmax"
# start: initialize on start;
# on_demand: only when needed (on simulation);
# null: never (only use pre-computed)
init_heuristics = "on_demand"
# none: don't normalize
# horizon: divide by horizon
# max: divide by max value found in dataset
heuristic_normalization = "none"
