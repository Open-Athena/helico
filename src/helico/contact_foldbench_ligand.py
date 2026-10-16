"""Public-API equivalent of OST 2.8 ligand scoring for a CLI reporting defect.

The official CLI incorrectly serializes ``inconsistent_residues`` (pairs of
residues) as individual residues. This wrapper uses the same public scorers,
mmCIF loading, hydrogen removal and default settings. It never modifies OST.
Only score serialization is reduced to the fields consumed by FoldBench.
"""
import argparse
import json
from pathlib import Path


def residue_name(residue):
    number = residue.number
    return f'{residue.chain.name}.{number.num}.{number.ins_code.strip(chr(0))}'


def score(model_path, reference_path):
    import ost
    from ost import io
    from ost.mol.alg.ligand_scoring_lddtpli import LDDTPLIScorer
    from ost.mol.alg.ligand_scoring_scrmsd import SCRMSDScorer

    if not ost.conop.GetDefaultLib():
        raise RuntimeError('OpenStructure compound library is required')

    def load(path):
        entity = io.LoadMMCIF(str(path), fault_tolerant=True)
        if not len(entity.residues):
            raise ValueError('Empty mmCIF: ' + str(path))
        entity.SetName(str(path))
        return ost.mol.CreateEntityFromView(entity.Select('ele != H and ele != D'),
                                            include_exlusive_atoms=False)

    reference, model = load(reference_path), load(model_path)
    shared = dict(model_ligands=None, target_ligands=None, resnum_alignments=False,
                  rename_ligand_chain=True, substructure_match=False, coverage_delta=.2)
    scorers = {
        'lddt_pli': LDDTPLIScorer(model, reference, lddt_pli_radius=6.,
                                add_mdl_contacts=False, **shared),
        'rmsd': SCRMSDScorer(model, reference, bs_radius=4., lddt_lp_radius=15., **shared),
    }
    result = dict(status='SUCCESS', backend='OST 2.8 public API; CLI metadata serialization recovery')
    for name, scorer in scorers.items():
        rows = []
        for target, predicted in scorer.assignment:
            row = dict(score=float(scorer.score_matrix[target, predicted]),
                       coverage=float(scorer.coverage_matrix[target, predicted]),
                       reference_ligand=residue_name(scorer.target_ligands[target]),
                       model_ligand=residue_name(scorer.model_ligands[predicted]))
            aux = scorer.aux_matrix[target, predicted]
            if name == 'rmsd':
                row['lddt_lp'] = float(aux['lddt_lp'])
                row['inconsistent_residues'] = [[residue_name(a), residue_name(b)]
                                               for a, b in aux['inconsistent_residues']]
            else:
                row['lddt_pli_n_contacts'] = int(aux['lddt_pli_n_contacts'])
            rows.append(row)
        result[name] = dict(assigned_scores=rows,
            model_ligand_unassigned_reason={residue_name(scorer.model_ligands[i]):
                scorer.guess_model_ligand_unassigned_reason(i) for i in scorer.unassigned_model_ligands},
            reference_ligand_unassigned_reason={residue_name(scorer.target_ligands[i]):
                scorer.guess_target_ligand_unassigned_reason(i) for i in scorer.unassigned_target_ligands})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(score(args.model, args.reference), indent=2) + '\n')
