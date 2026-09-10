import tensorflow as tf
from symnet3.unet import GATConvLayer, GATConvLayerDistance
from symnet3.action_decoder import ActionDecoder, get_activation_fn
import numpy as np
import pdb

symnet_params = {"channels", "num_postprocess", "num_preprocess", "attn_heads", "dropout_rate", "conv_type", "num_edge_types", "use_shared_gat"}
gat_params = {"channels", "attn_heads", "dropout_rate", "conv_type", "num_edge_types", "use_shared_gat"}

class SymNet3(tf.keras.Model):
    def __init__(self, general_params, se_params, ad_params, ge_params, tm_params):

        super(SymNet3, self).__init__()

        self.general_params = general_params
        self.se_params = se_params
        self.ad_params = ad_params
        self.ge_params = ge_params
        self.tm_params = tm_params
        self.tfm_params = self.ge_params["tfm_params"]

        self.out_deg = self.general_params["add_out_deg"]
        self.in_deg = self.general_params["add_in_deg"]
        self.bet_cen = self.general_params["add_bet_cen"]
        self.dist_leaves = self.general_params["add_dist_leaves"]
        self.use_bidir_edges = self.general_params["use_bidir_edges"]
        self.make_grid = self.general_params["make_grid"]
        self.remove_attn = self.general_params["remove_attn"]
        self.use_self_loops_in_all_adj = self.general_params["use_self_loops_in_all_adj"]
        self.num_heuristics = self.general_params["num_heuristics"]
        self.remove_dbn = self.general_params["remove_dbn"]

        self.use_distance_mat = se_params["use_distance_mat"]
        self.se_type = self.se_params["type"]
        self.se_count = self.se_params["num_se"]
        self.se_params["num_edge_types"] = None
        self.use_edge_types = se_params["use_edge_types"]
        self.preprocess_gat = se_params["use_preprocess_layer"]

        if self.use_distance_mat:
            arg_dict = dict((k, self.se_params[k]) for k in gat_params)
            arg_dict["num_edge_types"] = 1 # arg_dict["dbn_edge_types_to_idx"]
            arg_dict["filter_size"] = 1
            arg_dict["concat_last_gat"] = False
            arg_dict["attn_heads"] = se_params["num_dist_attn_heads"]
            arg_dict["return_attn_coef"] = True
            arg_dict["activation"] = get_activation_fn(self.se_params["activation"])
            self.gat_distance_mat = GATConvLayerDistance(**arg_dict)
            print("Built gat_distance_mat")

        if self.use_edge_types:
            self.se_params["num_edge_types"] = self.se_params["num_se"]
        
        if self.preprocess_gat:
            # Input size: number of fluents
            self.se_list_preprocess = self.create_state_encoders(self.se_params["num_preprocess"])
        # Input size: num_preprocess OR number of fluents
        self.se_list_postprocess = self.create_state_encoders(self.se_params["num_postprocess"])
        
        # Input size: num_postprocess
        self.final_node_embedder = self.create_final_encoder(self.se_params["out_dim"])

        self.ge_type = self.ge_params["type"]
        
        self.node_embed_dim = self.se_params["out_dim"]
        self.action_decoders = self.create_action_decoders()

        self.trained_steps = tf.Variable(0, trainable=False) 
        self.trained_epochs = tf.Variable(0, trainable=False) 


    def get_ckpt_parts(self):
        ckpt_parts = {}
        ckpt_parts["se_list"] = self.se_list_postprocess
        ckpt_parts["final_node_embedder"] = self.final_node_embedder
        ckpt_parts["next_state_projection"] = self.next_state_projection
        ckpt_parts["reward_projection"] = self.reward_projection
        ckpt_parts["action_decoders"] = self.action_decoders
        return ckpt_parts

    def init_network(self, env_wrapper):
        initial_state, _ = env_wrapper.reset()  # Initial state
        self.policy_prediction(states=[initial_state], env_wrapper=env_wrapper)

    def create_final_encoder(self, out_dim):
        activation_fn = get_activation_fn(self.se_params['activation'])
        return tf.keras.layers.Dense(units=out_dim, activation=activation_fn)

    def create_state_encoders(self, num_se):
        se_list = []
        activation_fn = get_activation_fn(self.se_params['activation'])
        if self.use_edge_types:
            args = dict((k, self.se_params[k]) for k in gat_params)
            args['filter_size'] = num_se
            args['activation'] = activation_fn
            print(f"Building a GAT with depth {num_se}")
            se_list.append(GATConvLayer(**args))
        else:
            for _ in range(num_se):
                args = dict((k, self.se_params[k]) for k in gat_params)
                args['activation'] = activation_fn
                se_list.append(GATConvLayer(**dict((k, self.se_params[k]) for k in symnet_params)))
        return se_list

    def create_action_decoders(self):
        action_decoders = []
        for _ in range(self.ad_params["num_action_templates"]):
            action_decoders.append(ActionDecoder(self.ad_params))
        return action_decoders

    def get_state_features(self, states: list, env_wrapper):
        adjacency_matrix = env_wrapper.get_processed_adj_mat(len(states))
        node_features = env_wrapper.get_processed_input(states)
        graph_features = env_wrapper.get_processed_graph_input(states)

        if self.use_bidir_edges:
            adjacency_matrix = adjacency_matrix + tf.transpose(adjacency_matrix, perm=[0, 1, 3, 2])
            adjacency_matrix = tf.clip_by_value(adjacency_matrix, 0, 1)

        return adjacency_matrix, node_features, graph_features

    def encode_nodes(self, encoders, node_features, adjacency_matrix):
        # se_embed_l: number of filters X number of states in the batch X number of features per node
        se_embed_l = []
        if self.use_edge_types:
            res = encoders[0](node_features, adjacency_matrix, self.use_self_loops_in_all_adj, self.remove_attn)
            se_embed_l.append(res)
        else:
            for i, se in enumerate(encoders):
                res = se(node_features, adjacency_matrix[i], self.use_self_loops_in_all_adj, self.remove_attn)
                se_embed_l.append(res)
        node_features = tf.concat(se_embed_l, axis=-1) # Number of node features is extended
        return node_features, se_embed_l

    def get_global_embedding(self, node_features, graph_features, adjacency_matrix):
        batch_size = node_features.shape[0]
        global_embed_pooled = tf.reduce_max(node_features, axis=1)
        global_embed = tf.reshape(tf.concat([global_embed_pooled, graph_features], axis=1), [batch_size, -1])
        if self.ge_type == "deep_global_pool":
            A = tf.reduce_max(adjacency_matrix, 0)
            global_embed_pooled_deep = self.global_embedder_net(node_features, A)
            global_embed = tf.concat([global_embed, global_embed_pooled_deep], axis=-1)
        return global_embed

    def filter_nodes(self, batch_size, nodes, node_embedding, extend=[]):
        # Select embeddings of nodes
        filtered = [tf.reshape(node_embedding[:, node, :], [batch_size, self.node_embed_dim]) for node in nodes]
        # Number of affected nodes X number of states in the batch X node embed size 
        return tf.concat(filtered + extend, axis=1)

    def get_action_scores(self, states, env_wrapper, node_embedding, global_features, training):
        action_details = env_wrapper.get_action_details()
        action_scores = [0 for i in range(len(action_details))]  # Score of each action
        action_affects = env_wrapper.get_action_affects()
        batch_size = len(states)
        for i in range(len(action_details)):
            action_template = action_details[i][0] # Name of lifted action
            eff_nodes = list(action_details[i][1]) # Effect nodes
            arg_nodes = action_details[i][2] # Parameter nodes

            global_embed = global_features
            if self.num_heuristics > 0:
                h = env_wrapper.estimate_successor_heuristics(states, i)
                global_embed = tf.concat([global_embed, h], axis=-1)

            arg_node_embed = global_embed
            if len(arg_nodes) == 0:  # Unparametrized action
                global_embed = None
            elif len(eff_nodes) > 0:
                # Number of affected nodes X number of states in the batch X node embed size 
                eff_node_embedding = self.filter_nodes(batch_size, eff_nodes, node_embedding)
                # Number of states in the batch X number of affected nodes X node embed size 
                eff_node_embedding = tf.reshape(eff_node_embedding, [batch_size, len(eff_nodes), self.node_embed_dim])
                # Max Pool: Number of states in the batch X 1 X node embed size 
                eff_embed_pooled = tf.reduce_max(eff_node_embedding, axis=1)
                # Number of states in the batch X node embed size 
                eff_embed_pooled = tf.reshape(eff_embed_pooled, [batch_size, self.node_embed_dim]) 
                # Number of arg nodes X number of states in the batch X node embed size 
                arg_node_embed = self.filter_nodes(batch_size, arg_nodes, node_embedding, [eff_embed_pooled])
            elif not self.remove_dbn: # Select embeddings of nodes
                # Wildfire case; Treat as NOOP
                if action_affects[action_template]:
                    # IF wildfire
                    action_template = action_details[0][0]
                    global_embed = None
                else:
                    padding = tf.zeros([batch_size, self.node_embed_dim], tf.float64)
                    #arg_embedding_list = [tf.reshape(node_embedding[:, inp, :], [batch_size, self.node_embed_dim]) for inp in arg_nodes]
                    #arg_embed = tf.concat(arg_embedding_list + [padding], tf.float64)], axis= 1)
                    arg_node_embed = self.filter_nodes(batch_size, arg_nodes, node_embedding, [padding])
            else:
                arg_node_embed = self.filter_nodes(batch_size, arg_nodes, node_embedding)

            action_scores[i] = self.action_decoders[action_template]([arg_node_embed, global_embed, training])

        return tf.concat(action_scores, axis=-1)

    def policy_prediction(self, states, env_wrapper, sample=False, training=True, prune_actions=False, return_attn_coef=False, return_node_emb=False):
        # node_features: number of states in the batch X number of nodes per state X number of node fluents
        # graph_features: number of states in the batch X number of state fluents
        adjacency_matrix, node_features, graph_features = self.get_state_features(states, env_wrapper)
        adjacency_matrix = np.transpose(adjacency_matrix, [0, 1, 3, 2])
        batch_size = node_features.shape[0]

        if self.preprocess_gat:
            node_features, _ = self.encode_nodes(self.se_list_preprocess, node_features, adjacency_matrix)
            
        if self.use_distance_mat:
            d = np.max(adjacency_matrix.astype("int32"), 0)
            d = env_wrapper.get_distance_mat(d)
            mask = env_wrapper.get_distance_mask()[None,:] # A 2D mask 
            mask = tf.repeat(mask, d.shape[0], axis=0)
            d = np.transpose(d, [1, 0, 2, 3])
            adjacency_matrix_fc = np.ones_like(d)
            distance_features, dist_attn_coef = self.gat_distance_mat(
                node_features, adjacency_matrix_fc, d, mask, 
                self.use_self_loops_in_all_adj, self.remove_attn, beta=1.0)
            node_features = tf.concat([node_features, distance_features], axis=-1)

        node_features, se_embed_l = self.encode_nodes(self.se_list_postprocess, node_features, adjacency_matrix)

        # final: number of filters X number of states in the batch X number of features per node
        node_features = self.final_node_embedder(node_features)
        global_features = self.get_global_embedding(node_features, graph_features, adjacency_matrix)

        # scores: number of states X number of grounded actions
        action_scores = self.get_action_scores(states, env_wrapper, node_features, global_features, training=training)
        
        if sample:
            logits = tf.nn.log_softmax(action_scores)
            if prune_actions:
                # Get the actions you want to keep
                masks = env_wrapper.get_prune_mask(states)
                masks = logits.dtype.min * (1.0 - masks)
                logits += masks
            probs = tf.random.categorical(logits=logits, num_samples=len(states), dtype=tf.int32)  # Return sampled actions
        else:
            if prune_actions:
                # Get the actions you want to keep
                masks = env_wrapper.get_prune_mask(states)
                masks = -10e9 * (1.0 - masks)
                action_scores += masks
            probs = tf.nn.softmax(action_scores)  # Expected shape is (batch_size,num_actions)

        if return_attn_coef:
            return probs, dist_attn_coef
        elif return_node_emb:
            return probs, se_embed_l
        else:
            return probs
