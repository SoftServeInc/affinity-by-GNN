import glob
import re
import torch
from dgl import DGLGraph
import numpy as np


class ProteinGraphMaker:
    """ 'Abstract' class describing the protein-to-graph converter; its derived classes
        are suppsoed to implement methods for parsing the specific input files (be it plain FASTA
        sequence, VPLG program outputs or whatever) and building DGLGraph as a result, as well
        as auxiliary methods for maintaining cache folder / best_model file naming conventions.
        For parsing, the implemented method should take pdb-id as the input and manage by itself
        all operations needed to find/load all paths/files/folder.
    """
    def __init__(self):
        typical_aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', # 12
                      'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] # 8
        self.typical_aminoacids = {k: i for i,k in enumerate(typical_aa)}

    def get_graph_cache_folder(self):
        return "Graph_Cache/"

    def get_best_model_filename(self):
        return 'affinity_best.pth'

    def pdbId_to_graph(self, pdbId):
        return None

    def _append_edge(self, proteinEdges, proteinEdgeFeatures, jA, jB, feature_vec):
        " a simple method added just to keep double-adding consistent"
        # Note:
        # Edges in DGLGraph are directed by default -- https://docs.dgl.ai/tutorials/models/3_generative_model/5_dgmg.html?highlight=undirected
        # DGLGraph is always directed. & In converting an undirected NetworkX graph into a DGLGraph, DGL internally
        # converts undirected edges to two directed edges -- https://docs.dgl.ai/guide/graph-external.html?highlight=undirected%20edges
        #
        # Note further that it might not seem nice to add each node twice here, but such choice will simplify
        # the below code for inserting edge features into the graph
        proteinEdges.append( (jA, jB) )
        proteinEdges.append( (jB, jA) )

        # add twice, to be consistent with (a,b),(b,a) duplication of edges in proteinEdges
        proteinEdgeFeatures.append( feature_vec )
        proteinEdgeFeatures.append( feature_vec )

    def lists_to_graph(self, node_features, proteinEdges, proteinEdgeFeatures):
        # convert all the accumulated data into a graph
        proteinGraph = DGLGraph()
        nNodes = len(node_features)

        # Nodes:
        proteinGraph.add_nodes( nNodes )
        proteinGraph.ndata['h'] = torch.Tensor(node_features)

        # Edges:
        for i,j in proteinEdges:
            proteinGraph.add_edge(i,j)
        #
        proteinGraph.edata['e'] = torch.Tensor(proteinEdgeFeatures)

        return proteinGraph


class DSSP_loader(ProteinGraphMaker):
    def __init__(self,
                 dssp_files_path,
                 includeAminoacidPhyschemFeatures = True,
                 cache_dir_prefix = ""):
        #
        super().__init__()

        self.dssp_path = dssp_files_path

        _all_letters_upper = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
                              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        _all_letters_lower = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                              'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

        self.dssp_categorial_node_props_inv = {
            'aminoacid': [
                          'V', 'M', 'D', 'N', 'R', 'L', 'I', 'A', 'T', 'G', 'E', 'Y', 'K', 'P', 'S', 'F', # 16 letters
                          'C', 'H', 'Q', 'W', #
                          'X', 'Z' # 'Z' is used by DSSP 4.0, but not by CMBI DSSP
                          # +see an ad hoc tricj to include lowercase letters is done below
                         ],
            'sse_class': ['', 'T', 'B', 'S', 'H', 'E', 'I', 'G', 'P'], # 'P' is used by original DSSP 4.0, but not by CMBI DSSP
            '3_turns_helix': ['', '3', '<', '>', 'X'],
            '4_turns_helix': ['', '4', '<', '>', 'X'],
            '5_turns_helix': ['', '5', '<', '>', 'X'],
            'geometrical_bend': ['', 'S'],
            'chirality': ['', '+', '-'],
            'beta_bridge_label_1': [''] +  _all_letters_upper + _all_letters_lower,
            'beta_bridge_label_2': [''] +  _all_letters_upper + _all_letters_lower,
            'beta_bridge_sheet_label': [''] +  _all_letters_upper
        } # actually, the inverse of it will be used to build features

        # Convert it to property->str->one-hot-index map:
        _map = {}
        for k,v in self.dssp_categorial_node_props_inv.items():
            _map[k] = { vv: i for i,vv in enumerate(v) }
            _map[k][None] = len(v)

        # map all 'unusual' aminoacids to 'others'
        _map['aminoacid'][None] += 1
        idxOtherAA = len(self.dssp_categorial_node_props_inv['aminoacid']) # the one-hot index for 'others'
        for x in _all_letters_lower:
            _map['aminoacid'][x] = idxOtherAA
        #
        self.dssp_categorial_node_props = _map

        self.featureSetPrefix = 'noPCFAA'

        self.cache_dir_prefix = cache_dir_prefix
    #

    #override
    def get_graph_cache_folder(self):
        return '%s_dssp_%s/' % (self.cache_dir_prefix, self.featureSetPrefix)

    #override
    def get_best_model_filename(self):
        return '%s_best_model_dssp_%s.pth' % (self.cache_dir_prefix, self.featureSetPrefix)

    def _parse_dssp_line(self, line):
        """ Parses a single line in the main section of .dssp file, and returns extracted
            data as dict
        """
        #  0+        10+       20+       30+     |  40+       50+       60+       70+       80+      90
        # 012345678901234567890123456789012345678|9012345678901234567890123456789012345678901234567890
        #'    5    6 A F  E     -ab  36 227A   0 |   221,-1.6   223,-1.3    30,-0.2     2,-0.5  -0.987'
        # 90+       100+      110+      120+      130+ |     140+      150+      160+      170+     180
        # 123456789012345678901234567890123456789012345|678901234567890123456789012345678901234567890
        #'  14.3-175.8-133.0 122.3   96.3   35.3   60.4|                A         A          5       '
        # 180+      190+      200+      210+      220+      230+      240+      250+     260
        # 12345678901234567890123456789012345678901234567890123456789012345678901234567890
        #'   6         36        227        221        223         30          2'
        r = {}
        r['dssp_res_num'] = int(line[:5]) # DSSP's sequential residue number, including _chain breaks_ (TODO)
        #r['pdb_res_num'] = int(line[5:11]) # crystallographers' 'residue sequence number'-given for reference only!
        r['chain_id'] = line[11].strip() # make chain breaks be ''
        r['aminoacid'] = line[13] # one letter amino acid code, lower case for SS-bridge CYS.
        r['sse_class'] = line[16].strip() # TODO: line[17]=='*' when line[16]=='!' (chain break)
        # line[16]=='!' - chain break residue detected as a discontinuity of backbone coordinates,
        # DSSP also detects a discontinuity in the PDB-supplied chain identifier, recorded as '*'==line[17]
        # // https://swift.cmbi.umcn.nl/gv/dssp/HTML/descrip.html

        # TODO: dict/indices for the following 'secondary structure summary' (based on columns 19-38)
        r['3_turns_helix'] = line[18].strip() # make it '' if empty (' ')
        r['4_turns_helix'] = line[19].strip()
        r['5_turns_helix'] = line[20].strip()
        r['geometrical_bend'] = line[21].strip()
        r['chirality'] = line[22].strip()
        r['beta_bridge_label_1'] = line[23].strip()
        r['beta_bridge_label_2'] = line[24].strip()

        # residue number of first and second bridge partner followed by one letter sheet label:
        r['beta_bridge_partner_resnum_1'] = int(line[25:29])
        r['beta_bridge_partner_resnum_2'] = int(line[29:33])
        r['beta_bridge_sheet_label'] = line[33].strip()

        r['num_waters'] = int(line[34:38]) # number of water molecules in contact with this residue *10,
        # or residue water exposed surface in Angstrom**2.

        # Hydrogen bonding data:
        for w, offs in zip(['first_NH_O', 'first_O_HN', 'second_NH_O', 'second_O_HN'], [0,50-39,61-39,72-39]):
            r['%s_idx' % w] = int(line[offs+39 : offs+45])
            r['%s_E' % w] = float(line[offs+46 : offs+50])
        #
        r['cos_CO_CO'] = float(line[83:91]) # TCO - cosine of angle between C=O of residue I and C=O of residue I-1
        r['kappa_bend_angle'] = float(line[91:97]) #  KAPPA - bend angle, defined by C-alpha of residues I-2,I,I+2
        r['alpha_torsion'] = float(line[97:103]) #ALPHA - torsion angle, defined by C-alpha atoms
                           # of residues I-1,I,I+1,I+2. Used to define chirality (structure code '+' or '-').
        r['phi'] = float(line[103:109]) # IUPAC peptide backbone torsion angles
        r['psi'] = float(line[109:115])
        r['r_Calpha'] = np.array([float(line[i:i+7]) for i in [115, 122, 129]]) # C-alpha atom coordinates

        return r

    def _parse_dssp(self, fname):
        # note that the lines in dssp files seem to consist of fixed-width fields, so that, e.g.,
        # ' 27.5-102.6' is perfectly valid pair of floats
        templ_totCount = 'TOTAL NUMBER OF RESIDUES, NUMBER OF CHAINS, NUMBER OF ' + \
                         'SS-BRIDGES(TOTAL,INTRACHAIN,INTERCHAIN)'
        templ_mainHdr = 'RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    ' + \
                        'TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA'
        nTotResidues = None
        #nChains = None
        active_section = None
        chains = {} # chainId -> {'property': list_of_values_for_all_residues }
        chains2 = {} # chainId -> [list of {property: value} dicts for all residues ]
        #
        with open(fname) as f:
            for line in f:
                line = line.rstrip()
                if line.find(templ_totCount) != -1:
                    nTotResidues = int(line[:5])
                    #nChains = int(line[5:8]) # when chain breaks are present, this will
                                                # count 'pieces', not 'chains'
                if active_section == 'main':
                    dat = self._parse_dssp_line(line)
                    chainId = dat['chain_id']
                    if len(chainId) > 0: # if not chain break record
                        if chainId not in chains:
                            chains[chainId] = {}
                            chains2[chainId] = []
                        del dat['chain_id']
                        for k,v in dat.items():
                            if k not in chains[chainId]:
                                chains[chainId][k] = []
                            chains[chainId][k].append(v)
                        #
                        chains2[chainId].append(dat)
                    #
                if line.find(templ_mainHdr) != -1:
                    active_section = 'main'
            #
        #
        # because of numbering chain breaks as a separate entities, we can not do something line
        #assert len(chains.keys()) == nChains, ('inconsistent number of chains in', fname)
        return chains, chains2

    def explore_property_values(self, pdbId, dest_dict):
        """ Merely parses .dssp files and updates dest_dict with possible
            str-type values of the residue properties
        """
        assert type(dest_dict) is dict
        #
        chains, _ = self._parse_dssp(self.dssp_path + pdbId + '.cif.dssp')
        for chainId, chain in chains.items():
            for prop_name, prop_list in chain.items():
                # len(prop_list)>0 always - otherwise, prop_name is not in this dict
                for p in prop_list:
                    if type(p) is str:
                        if prop_name not in dest_dict:
                            dest_dict[prop_name] = {}
                        if p not in dest_dict[prop_name]:
                            dest_dict[prop_name][p] = 0
                        dest_dict[prop_name][p] += 1
        return dest_dict # for convenience only; dest_dict has already been updated
    #

    def _build_node_features_vec(self, rec):
        self.ignored_props = ['dssp_res_num',
                              'beta_bridge_label_1', 'beta_bridge_label_2', 'beta_bridge_sheet_label',
                              'first_NH_O_idx', 'first_O_HN_idx', 'second_NH_O_idx', 'second_O_HN_idx',
                              'first_NH_O_E', 'first_O_HN_E', 'second_NH_O_E', 'second_O_HN_E',
                              'beta_bridge_partner_resnum_1',  'beta_bridge_partner_resnum_2',
                              'alpha_torsion',  'phi', 'psi', 'r_Calpha'
                             ]
        #self.int_and_flaot_props = [ 'num_waters', 'cos_CO_CO', 'kappa_bend_angle' ] #TODO: should cos_CO_CO be here?
        result = []
        # one-hot encodings for categorial properties:
        #for prop_name, propVal2idx in self.dssp_categorial_node_props.items():
        for prop_name in ['aminoacid', 'sse_class']:
            propVal2idx = self.dssp_categorial_node_props[prop_name]

            vec = [0] * propVal2idx[None]
            str_prop = rec[prop_name]
            prop_idx = propVal2idx[str_prop]
            vec[ prop_idx ] = 1
            result += vec

        return result

    def pdbId_to_graph(self, pdbId):
        """ The key method to read-in fasta from file associated with pdbId,
            parse it, convert to a graph and return the obtained DGLGraph
        """

        node_features = []
        proteinEdges = []
        proteinEdgeFeatures = []

        _, chains2 = self._parse_dssp(self.dssp_path + pdbId + '.cif.dssp')

        resnum2idx = {} # dsspResNum -> indexOfNodeIn_node_features; will be needed to make edges for H-bonds

        _pre_h_bonds = [] # list of edges: [(nodeFrom, nodeTo, h_bond_type ('NH_O' or 'O_HN'), energy ), ...]

        num_edge_features = 5 # not including the 0-th feature: 1==peptide, 0==h-bond
        # E_hb; alpha, phi, psi, reserved

        for chainId, theChain in chains2.items():
            for iResidue, residue_rec in enumerate(theChain):
                ftrs = self._build_node_features_vec(residue_rec)
                node_features.append( ftrs )
                iDsspResnum = residue_rec['dssp_res_num']
                resnum2idx[iDsspResnum] = len(node_features)-1
                # TODO: check recipr.
                for ik, k in enumerate(['first_NH_O', 'first_O_HN', 'second_NH_O', 'second_O_HN']):
                    jj = residue_rec[k+'_idx']
                    hb_E = residue_rec[k+'_E']
                    if jj != 0:
                        # here, we only save this adge, but don't add it to the graph yet!
                        _pre_h_bonds.append( (iDsspResnum, iDsspResnum + jj, k[-4:], hb_E ) )

                # edges describing the peptide bonds (within the current chain only!):
                if iResidue >= 1:
                    I,J = len(node_features)-1, len(node_features)-2 # 'global' node numbers, so to say
                    edge_feature_vec = [1] + [0]*num_edge_features
                                      # ^+- this first '1' distinguishes peptide bond from h-bond
                    self._append_edge(proteinEdges, proteinEdgeFeatures, I, J, edge_feature_vec)
            #
        #

        uniq_hbonds = {}
        for i,j,hb_type,E_hb in _pre_h_bonds:
            ij = (i,j) if i<j else (j,i)
            if ij not in uniq_hbonds:
                uniq_hbonds[ij] = []
            uniq_hbonds[ij].append( E_hb )
        for (i,j), E_hbs in uniq_hbonds.items():
            # create only a single edge connecting these residues, and set its energy to sum of HB energies
            # in this way we'll characterize the total 'strength' of the edge
            edge_feature_vec = [0] + [0]*num_edge_features
                              # ^+- this first '0' distinguishes hydrogen bond from peptide bond
            edge_feature_vec[1] = np.sum(E_hbs)
            assert i in resnum2idx, (pdbId, i, 'not in resnum2idx')
            assert j in resnum2idx, (pdbId, j, 'not in resnum2idx')
            I,J = resnum2idx[i], resnum2idx[j]
            self._append_edge(proteinEdges, proteinEdgeFeatures, I, J, edge_feature_vec)
        # now just convert all the accumulated data into a graph
        return self.lists_to_graph(node_features, proteinEdges, proteinEdgeFeatures)
    #
#
