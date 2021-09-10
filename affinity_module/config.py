from dgllife.utils import smiles_to_complete_graph, WeaveAtomFeaturizer, WeaveEdgeFeaturizer
from functools import partial


current_config = {
    'random_seed': 8,
    'mlp_hidden_layer': 256,
    'smiles_num_layers': 2,
    'smiles_num_timesteps': 2,
    'smiles_graph_feat_size': 200,
    'fasta_num_layers': 2,
    'fasta_num_timesteps': 2,
    'fasta_graph_feat_size': 200,
    'n_tasks': 1,
    'dropout': 0.05,
    'weight_decay': 10 ** (-5.0),
    'lr': 0.001,
    'batch_size': 32,
    'num_epochs': 1000,
    'frac_train': 0.8,
    'frac_val': 0.1,
    'frac_test': 0.1,
    'patience': 50,
    'metric_name': 'mae',
    'model': 'mhGANN',
    'mode': 'lower',
    'smiles_to_graph': partial(smiles_to_complete_graph, add_self_loop=True),
    'smiles_node_featurizer': WeaveAtomFeaturizer(),
    'smiles_edge_featurizer': WeaveEdgeFeaturizer(max_distance=2),
    'load_checkpoint': False,
    'inference': False,
    'argv_valFold': 4,
    'argv_testFold': 4,
    'cache_dir_prefix': 'graph_cache',
    'dssp_files_path': 'example_data/dssp/',
    'master_data_table': "example_data/input.csv"
}

def get_config():
    return current_config
