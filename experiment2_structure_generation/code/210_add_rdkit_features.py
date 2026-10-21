import argparse
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors


_DEFAULT_INPUT = (
    "work/predicted_formula_multihyp_v1/"
    "cross_formula/xgb_oof_v1/"
    "rank_features.jsonl"
)

_DEFAULT_OUTPUT = (
    "work/predicted_formula_multihyp_v1/"
    "cross_formula/xgb_oof_v1/"
    "rank_features_rdkit.jsonl"
)


ap = argparse.ArgumentParser(
    description="Append RDKit descriptor features to rank_features.jsonl"
)
ap.add_argument("--input", default=_DEFAULT_INPUT, help="input rank_features.jsonl")
ap.add_argument("--output", default=_DEFAULT_OUTPUT, help="output rank_features_rdkit.jsonl")
args = ap.parse_args()

INPUT = args.input
OUTPUT = args.output


def get_rdkit_features(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "rdkit_invalid":1.0
        }


    return {

        # size
        "mol_weight":
            Descriptors.MolWt(mol),

        "heavy_atom_count":
            mol.GetNumHeavyAtoms(),

        "atom_count":
            mol.GetNumAtoms(),


        # topology
        "ring_count":
            rdMolDescriptors.CalcNumRings(mol),

        "aromatic_ring_count":
            rdMolDescriptors.CalcNumAromaticRings(mol),

        "aliphatic_ring_count":
            rdMolDescriptors.CalcNumAliphaticRings(mol),


        # chemistry
        "hba":
            Lipinski.NumHAcceptors(mol),

        "hbd":
            Lipinski.NumHDonors(mol),

        "tpsa":
            rdMolDescriptors.CalcTPSA(mol),

        "rotatable_bonds":
            Lipinski.NumRotatableBonds(mol),

        "fraction_csp3":
            rdMolDescriptors.CalcFractionCSP3(mol),


        # complexity
        "chiral_centers":
            len(Chem.FindMolChiralCenters(
                mol,
                includeUnassigned=True
            )),

        "bertz_complexity":
            Descriptors.BertzCT(mol),


        # smiles
        "smiles_length":
            len(smiles),

        "rdkit_invalid":0.0
    }



Path(OUTPUT).parent.mkdir(
    parents=True,
    exist_ok=True
)


count=0


with open(INPUT) as fin, open(OUTPUT,"w") as fout:

    for line in fin:

        x=json.loads(line)

        rdkit_feat=get_rdkit_features(
            x["smiles"]
        )


        x["features"].update(
            rdkit_feat
        )


        fout.write(
            json.dumps(
                x,
                ensure_ascii=False
            )+"\n"
        )

        count+=1


print("Done:",count)

print("Output:",OUTPUT)