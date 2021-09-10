from affinity_module.protein_graph_loaders import ProteinGraphMaker
from dgllife.utils import smiles_to_bigraph
from dgl.data.utils import save_graphs, load_graphs
from rdkit import Chem
import os
import dgl.backend as F
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
import json


class VPLGDataset(object):
    def __init__(self,
                 smiles_to_graph=smiles_to_bigraph,
                 smiles_node_featurizer=None,
                 smiles_edge_featurizer=None,
                 pdb_id_col_name="pdbId",
                 smiles_col_name="ligandSMILES",
                 target_col_name="logKx",
                 foldId_col_name=None, # if not None, will read and save foldId
                 master_data_table = "",
                 pdb2graph_translator = None, # must be pre-created class derived from ProteinGraphMaker
                 log_every=500,
                 cache_size=500,
                 load=False, # whether to load graphs from cache
                 inference=False,
                 append_itemId = False, # whether to return item id in __getitem__
                 columns_to_ignore = []
                ):

        assert isinstance(pdb2graph_translator, ProteinGraphMaker)
        self.pdb2graph_translator = pdb2graph_translator

        # Note that for now, we still re-read .csv file even if load is set to True; this is needed at
        # least to be able to handle fold_ids
        self.df = pd.read_csv(master_data_table)
        self.smiles = []
        self.smiles_graphs = []
        self.fasta_graphs = []
        self.labels = []

        cache_basePath = self.pdb2graph_translator.get_graph_cache_folder()

        self.smiles_graphs_cache_path = cache_basePath + "smiles_graphs/smiles_dglgraph"
        self.fasta_graphs_cache_path = cache_basePath + "fasta_graphs/fasta_dglgraph"
        self.smiles_cache_path = cache_basePath + "smiles/smiles"
        self.labels_cache_path = cache_basePath + "labels/label"

        self.cache_size = cache_size
        self.inference = inference
        self.append_itemId = append_itemId

        self._makedir(self.smiles_graphs_cache_path)
        self._makedir(self.fasta_graphs_cache_path)
        self._makedir(self.smiles_cache_path)
        self._makedir(self.labels_cache_path)

        if target_col_name is None:
            self.task_names = self.df.columns.drop([smiles_col_name, pdb_id_col_name]+columns_to_ignore).tolist()
        else:
            self.task_names = [target_col_name]
        #

        if foldId_col_name is not None:
            self._fold_list = self.df[foldId_col_name].values.tolist() # raw data, to be used to fill in self.foldIds
            self.foldIds = [] # will be filled in, and will contain the number of elems equal to num of VALID graphs
        else:
            self._fold_list = None
            self.foldIds = None
        #

        self.n_tasks = len(self.task_names)
        self._pre_process(smiles_to_graph, smiles_node_featurizer, smiles_edge_featurizer,
                          smiles_col_name, pdb_id_col_name, load, log_every)

    def _makedir(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer,
                     smiles_col_name, pdb_id_col_name, load, log_every):

        mis_options_file = os.path.join(os.path.dirname(self.fasta_graphs_cache_path), 'misc_options.json')

        smiles_graphs_cache_dir = os.path.dirname(self.smiles_graphs_cache_path)
        fasta_graphs_cache_dir = os.path.dirname(self.fasta_graphs_cache_path)
        smiles_cache_dir = os.path.dirname(self.smiles_cache_path)
        labels_dir = os.path.dirname(self.labels_cache_path)

        smiles_graphs_cache_list = sorted([f for f in os.listdir(smiles_graphs_cache_dir) if f.endswith('.bin')])
        fasta_graphs_cache_list = sorted([f for f in os.listdir(fasta_graphs_cache_dir) if f.endswith('.bin')])
        smiles_cache_list = sorted([f for f in os.listdir(smiles_cache_dir) if f.endswith('.bin')])

        if not self.inference:
            labels_list = sorted([f for f in os.listdir(labels_dir) if f.endswith('.bin')])

        if not load:
            # remove all previously existing graph cache files first!
            print('deleting old cache...')
            dir_list = [smiles_graphs_cache_dir, fasta_graphs_cache_dir, smiles_cache_dir]
            if not self.inference:
                dir_list.append(labels_dir)
            #
            for _dir in dir_list:
                flist = sorted([f for f in os.listdir(_dir) if f.endswith('.bin')])
                for fnm in flist:
                    os.remove(os.path.join(_dir, fnm))
                print('%s: %d deleted' % (_dir, len(flist)))
            #
        #

        if load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')

            if self.inference:
                for smiles_graphs_cache_file,\
                    fasta_graphs_cache_file,\
                    smiles_cache_file,\
                    fasta_cache_file in zip(smiles_graphs_cache_list,
                                             fasta_graphs_cache_list,
                                             smiles_cache_list,
                                             fasta_cache_list):
                    smiles_graphs_cache_file = os.path.join(smiles_graphs_cache_dir, smiles_graphs_cache_file)
                    fasta_graphs_cache_file = os.path.join(fasta_graphs_cache_dir, fasta_graphs_cache_file)
                    smiles_cache_file = os.path.join(smiles_cache_dir, smiles_cache_file)
                    fasta_cache_file = os.path.join(fasta_cache_dir, fasta_cache_file)

                    smiles_graphs, _ = load_graphs(smiles_graphs_cache_file)
                    fasta_graphs, _ = load_graphs(fasta_graphs_cache_file)

                    self.smiles_graphs.extend(smiles_graphs)
                    self.fasta_graphs.extend(fasta_graphs)

                    with open(smiles_cache_file, 'rb') as f:
                        self.smiles.extend(pickle.load(f))
            else:
                for smiles_graphs_cache_file,\
                    fasta_graphs_cache_file,\
                    smiles_cache_file,\
                    labels_cache_file in zip(smiles_graphs_cache_list, # [:32] to limit cache blocks
                                             fasta_graphs_cache_list, # [:32]
                                             smiles_cache_list, # [:32]
                                             labels_list): # [:32]
                    smiles_graphs_cache_file = os.path.join(smiles_graphs_cache_dir, smiles_graphs_cache_file)
                    fasta_graphs_cache_file = os.path.join(fasta_graphs_cache_dir, fasta_graphs_cache_file)
                    smiles_cache_file = os.path.join(smiles_cache_dir, smiles_cache_file)
                    labels_cache_file = os.path.join(labels_dir, labels_cache_file)

                    smiles_graphs, _ = load_graphs(smiles_graphs_cache_file)
                    fasta_graphs, _ = load_graphs(fasta_graphs_cache_file)

                    self.smiles_graphs.extend(smiles_graphs)
                    self.fasta_graphs.extend(fasta_graphs)

                    with open(smiles_cache_file, 'rb') as f:
                        self.smiles.extend(pickle.load(f))
                    with open(labels_cache_file, 'rb') as f:
                        self.labels.extend(pickle.load(f))

            # restore misc. flags&options
            with open(mis_options_file) as jf:
                dat = json.load(jf)
            self.foldIds = dat['foldIds'] # will return a list or None

            print('smiles:',        len(self.smiles))
            print('smiles_graphs:', len(self.smiles_graphs))
            print('fasta_graphs:',  len(self.fasta_graphs))
            if not self.inference:
                self.labels = F.zerocopy_from_numpy(np.nan_to_num(self.labels).astype(np.float32))
                print('labels:',        len(self.labels))
        else:
            j_offset = 0
            i_offset = 0 #1
            print('Processing dgl graphs from scratch...')
            smiles_raw = self.df[smiles_col_name].values.tolist()
            pdb_id = self.df[pdb_id_col_name].values.tolist()
            labels_raw = self.df[self.task_names].values.tolist()
            last = len(smiles_raw) - 1
            prev = 0
            j = 0

            print('...to load:', len(smiles_raw), len(pdb_id), len(labels_raw))

            for i, (s, f, l) in enumerate(zip(smiles_raw, pdb_id, labels_raw)):
                if (i + i_offset) % log_every == 0:
                    print('Processing graph {:d}/{:d}'.format(i+i_offset, len(self)))

                _loaded_ok = True
                try:
                    sg = smiles_to_graph(s, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
                except Exception as e:
                    _loaded_ok = False
                    print('Exception in smiles_to_graph in line: {}, for SMILES; {}, PDB_ID: {}'.format(i, s, f))
                    print(e)

                try:
                    fg = self.pdb2graph_translator.pdbId_to_graph(f)
                except Exception as e:
                    _loaded_ok = False
                    print('Exception in pdbId_to_graph in line: {}, for SMILES; {}, PDB_ID: {}'.format(i, s, f))
                    print(e)
                if len(fg.ndata['h'])==0:
                    _loaded_ok = False
                    print('Error: PDB_ID {} in line {} yields graph with no nodes'.format(f, i))                    

                try:
                    assert 'h' in fg.ndata, ('no nodes in graph obtained from', plg_filenames)
                    assert 'e' in fg.edata, ('no edges in graph obtained from', plg_filenames)
                    # skip iteration if any fbg keys except 'a', 'p', 'm' are present
                except Exception as e:
                    _loaded_ok = False
                    print('Exception no_h/no_e in line: {}, for SMILES; {}, PDB_ID: {}'.format(i, s, f))
                    print(e)

                if _loaded_ok:
                    self.smiles_graphs.append(sg)
                    self.fasta_graphs.append(fg)
                    self.smiles.append(s)

                    if self.foldIds is not None:
                        self.foldIds.append(self._fold_list[i]) # only for those graphs which have been loaded with no errors!

                    if not self.inference:
                        self.labels.append(l)
                    j += 1

                if (j + 1) % self.cache_size == 0 or i == last:
                    print('Caching graph {:d}/{:d}'.format(j+j_offset, len(self)))
                    save_graphs('{}_{:07d}.bin'.format(self.smiles_graphs_cache_path, j+j_offset+1), self.smiles_graphs[prev:j])
                    save_graphs('{}_{:07d}.bin'.format(self.fasta_graphs_cache_path, j+j_offset+1), self.fasta_graphs[prev:j])

                    with open('{}_{:07d}.bin'.format(self.smiles_cache_path, j+j_offset+1), 'wb') as f:
                        pickle.dump(self.smiles[prev:j], f)
                    if not self.inference:
                        with open('{}_{:07d}.bin'.format(self.labels_cache_path, j+j_offset+1), 'wb') as f:
                            pickle.dump(self.labels[prev:j], f)

                    # dump misc. flags&options
                    with open(mis_options_file, 'w') as jf:
                        json.dump({'foldIds': self.foldIds}, jf)

                    prev = j

            print('smiles:', len(self.smiles))
            print('smiles_graphs:',len(self.smiles_graphs))
            print('fasta_graphs:',len(self.fasta_graphs))
            if not self.inference:
                print('labels:',len(self.labels))
                # np.nan_to_num will also turn inf into a very large number
                self.labels = F.zerocopy_from_numpy(np.nan_to_num(self.labels).astype(np.float32))

    def __getitem__(self, item):
        if self.append_itemId:
            return self.fasta_graphs[item], self.smiles[item], self.smiles_graphs[item], item

        if self.inference:
            return self.fasta_graphs[item], self.smiles[item], self.smiles_graphs[item]
        else:
            return self.fasta_graphs[item], self.smiles[item], self.smiles_graphs[item], self.labels[item]
    #

    def __len__(self):
        """Size for the dataset
        Returns
        -------
        int
            Size for the dataset
        """
        return len(self.smiles)
    #
#


class FoldsOf_VPLGDataset:
    # Selects the items from source Dataset, accorging to conditions on fold and/or on graph size
    def __init__(self, src, folds = [], max_nodes = 7000):
        """ src - VPLGDataset to get data from; 
            folds - list of foldIds to be returned by this object, of [] to use all data
            max_nodes - int or function (returning True to use the graph, arg == graph)
        """
        assert isinstance(src, VPLGDataset)

        if callable(max_nodes): # type(max_nodes) is function:
            graph_filter = max_nodes
        else:
            # use the default condition
            def _default_filter_func(fg):
                if len(fg.ndata['h']) == 0:
                    print("Error: len(fg.ndata['h']) == 0")
                    return False
                #
                if max_nodes <= 0:
                    return True
                #

                res = len(fg.ndata['h']) <= max_nodes
                if not res:
                    print("skip: len(fg.ndata['h']) ==", len(fg.ndata['h']))
                return res
            #
            graph_filter = _default_filter_func
        #

        self.src = src
        self.raw_indices = []
        folds = set(folds)
        assert len(folds) >= 0
        if len(folds) > 0:
            assert src.foldIds is not None

        for j, item in enumerate(self.src):
            if len(folds) > 0:
                # first check, if fold is appropriate
                if src.foldIds[j] not in folds:
                    continue
            # of course, don't skip enything if len(folds) == 0 -- 'use all folds'

            # Now, check if the graph satisfies selection conditions;
            # item is tuple of either (fasta_graph, smiles, smiles_graph) or of
            # (fasta_graph, smiles, smiles_graph, label)
            if not graph_filter(item[0]):
                continue
            # else:
            self.raw_indices.append(j)
        #
    #
    def asDataLoader(self, **kwargs):
        return DataLoader(dataset=self, **kwargs)

    def __getitem__(self, item):
        return self.src[ self.raw_indices[item] ]

    def __len__(self):
        return len(self.raw_indices)
