import scipy.spatial
import sys
import os
import pickle
import torch

from rdkit import Chem

from data.data_prepare import RetroDiffDataset, RetroDiffDataInfos, RetroDiffExtraFeature
from arguments import arg_parses

from visualization.draw import MolecularVisualization

from smiles_utils import smi_tokenizer, clear_map_number, SmilesGraph
from smiles_utils import canonical_smiles, canonical_smiles_with_am, remove_am_without_canonical, \
    extract_relative_mapping, get_nonreactive_mask, randomize_smiles_with_am

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_synthon_from_product(args):
    dataset = RetroDiffDataset(dataset_name="USPTO50K",
                               train_batch_size=args.train_batch_size,
                               val_batch_size=args.val_batch_size,
                               test_batch_size=args.test_batch_size,
                               num_workers=args.num_workers,
                               )
    train_dataloader, val_dataloader, test_dataloader = dataset.prepare()
    
    DatasetInfos = RetroDiffDataInfos()
    input_dims, output_dims = DatasetInfos.compute_io_dims(val_dataloader)

    MolVis = MolecularVisualization(remove_h="False", dataset_infos=DatasetInfos)

    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch)

        p_smiles = batch.p_smiles[0]
        r_smiles = batch.r_smiles[0]

        p_atom = batch.p_atom_symbols[0].argmax(dim=-1)
        r_atom = batch.r_atom_symbols[0].argmax(dim=-1)

        p_bond = batch.p_bond_symbols[0].argmax(dim=-1)
        r_bond = batch.r_bond_symbols[0].argmax(dim=-1)

        p_adjs = batch.p_adjs[0]
        r_adjs = batch.r_adjs[0]

        # print(p_smiles)
        # print(r_smiles)

        legend_list = ["Product" + str(batch_idx) + ": " + p_smiles]
        out_filename = "product" + str(batch_idx) + ".png"
        MolVis.graph2mol(p_atom, p_bond, legend_list, out_filename)

        legend_list = ["Reactant: " + str(batch_idx) + ": " + r_smiles]
        out_filename = "reactant"  + str(batch_idx) + ".png"
        MolVis.graph2mol(r_atom, r_bond, legend_list, out_filename)

        if batch_idx > 10:
            break


def parse_smi(prod, reacts, react_class, build_vocab=False, randomize=False):
    ''' Process raw atom-mapped product and reactants into model-ready inputs
        :param prod: atom-mapped product
        :param reacts: atom-mapped reactants
        :param react_class: reaction class
        :param build_vocab: whether it is the stage of vocab building
        :param randomize: whether do random permutation of the reaction smiles
        :return:
    '''
    known_class = False

    intermediate_folder='./semi-template'
    vocab_file = 'vocab_share.pk'
    with open(os.path.join(intermediate_folder, vocab_file), 'rb') as f:
        src_itos, tgt_itos = pickle.load(f)
    src_stoi = {src_itos[i]: i for i in range(len(src_itos))}
    tgt_stoi = {tgt_itos[i]: i for i in range(len(tgt_itos))}
    
    # Process raw prod and reacts:
    cano_prod_am = canonical_smiles_with_am(prod)
    cano_reacts_am = canonical_smiles_with_am(reacts)

    cano_prod = clear_map_number(prod)
    cano_reacts = remove_am_without_canonical(cano_reacts_am)

    if build_vocab:
        return cano_prod, cano_reacts

    if Chem.MolFromSmiles(cano_reacts) is None:
        cano_reacts = clear_map_number(reacts)

    if Chem.MolFromSmiles(cano_prod) is None or Chem.MolFromSmiles(cano_reacts) is None:
        return None

    if randomize:
        # print('permute product')
        cano_prod_am = randomize_smiles_with_am(prod)
        cano_prod = remove_am_without_canonical(cano_prod_am)
        if np.random.rand() > 0.5:
            # print('permute reacts')
            cano_reacts_am = '.'.join(cano_reacts_am.split('.')[::-1])
            cano_reacts = remove_am_without_canonical(cano_reacts_am)

    # Get the smiles graph
    smiles_graph = SmilesGraph(cano_prod)
    # Get the nonreactive masking based on atom-mapping
    gt_nonreactive_mask = get_nonreactive_mask(cano_prod_am, prod, reacts, radius=1)
    # Get the context alignment based on atom-mapping
    position_mapping_list = extract_relative_mapping(cano_prod_am, cano_reacts_am)

    # Note: gt_context_attn.size(0) = tgt.size(0) - 1; attention for token that need to predict
    gt_context_attn = torch.zeros(
        (len(smi_tokenizer(cano_reacts_am)) + 1, len(smi_tokenizer(cano_prod_am)) + 1)).long()
    for i, j in position_mapping_list:
        gt_context_attn[i][j + 1] = 1

    # Prepare model inputs
    src_token = smi_tokenizer(cano_prod)
    tgt_token = ['<sos>'] + smi_tokenizer(cano_reacts) + ['<eos>']
    if known_class:
        src_token = [react_class] + src_token
    else:
        src_token = ['<UNK>'] + src_token
    gt_nonreactive_mask = [True] + gt_nonreactive_mask

    src_token = [src_stoi.get(st, src_stoi['<unk>']) for st in src_token]
    tgt_token = [tgt_stoi.get(tt, tgt_stoi['<unk>']) for tt in tgt_token]

    return src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask



if __name__ == "__main__":
    args = arg_parses()
    #get_synthon_from_product(args)
    prod = "[CH3:1][c:2]1[n:3][s:4][c:5]([NH:6][C:7](=[O:8])[CH2:9][c:10]2[cH:11][cH:12][c:13]([OH:14])[c:15]([NH2:16])[cH:17]2)[c:18]1[Cl:19]"
    react = "O=[N+:16]([O-])[c:15]1[c:13]([OH:14])[cH:12][cH:11][c:10]([CH2:9][C:7]([NH:6][c:5]2[s:4][n:3][c:2]([CH3:1])[c:18]2[Cl:19])=[O:8])[cH:17]1"
    react_class = 1

    src_token, smiles_graph, tgt_token, gt_context_attn, gt_nonreactive_mask = parse_smi(prod, react, react_class, build_vocab=False, randomize=False)
    # print(src_token)
    # print(smiles_graph)
    # print(tgt_token)
    # print(gt_context_attn)
    print(gt_nonreactive_mask)