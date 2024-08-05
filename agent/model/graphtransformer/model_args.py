
class ModelArgs():
    def __init__(self,
                num_classes: int,
                max_nodes: int,
                num_node_type: int,
                num_node_attr: int,
                num_in_degree: int,
                num_out_degree: int,
                num_edges: int,
                num_spatial: int,
                num_edge_dis: int,
                edge_type: str,
                multi_hop_max_dist: int,
                num_encoder_layers: int = 12,
                embedding_dim: int = 768,
                ffn_embedding_dim: int = 768,
                num_attention_heads: int = 32,
                dropout: float = 0.1,
                attention_dropout: float = 0.1,
                activation_dropout: float = 0.1,
                layerdrop: float = 0.0,
                encoder_normalize_before: bool = False,
                pre_layernorm: bool = False,
                apply_graphormer_init: bool = False,
                activation_fn: str = "gelu",
                embed_scale: float = None,
                freeze_embeddings: bool = False,
                n_trans_layers_to_freeze: int = 0,
                export: bool = False,
                traceable: bool = False,
                q_noise: float = 0.0,
                qn_block_size: int = 8):
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.num_node_type = num_node_type
        self.num_node_attr = num_node_attr
        self.num_in_degree = num_in_degree
        self.num_out_degree = num_out_degree
        self.num_edges = num_edges
        self.num_spatial = num_spatial
        self.num_edge_dis = num_edge_dis
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_encoder_layers = num_encoder_layers
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.encoder_normalize_before = encoder_normalize_before
        self.pre_layernorm = pre_layernorm
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn
        self.embed_scale = embed_scale
        self.freeze_embeddings = freeze_embeddings
        self.n_trans_layers_to_freeze = n_trans_layers_to_freeze
        self.export = export
        self.traceable = traceable
        self.q_noise = q_noise
        self.qn_block_size = qn_block_size
        