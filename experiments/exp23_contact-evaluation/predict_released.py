"""Reproduce unconditioned inference from a released weight file and prepared input."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from helico.contact_evaluate import load_payload, predict
from helico.model import Helico, HelicoConfig


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = HelicoConfig(**checkpoint['model_config'])
    config.n_diffusion_steps = 200
    config.n_cycles = 1
    config.msa_sample_cutoff = config.msa_sample_min_eval = 512
    model = Helico(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    # Keep parameters and sampler coordinates in FP32; predict() applies
    # BF16 autocast to the network while preserving rigid coordinate rotations.
    model = model.cuda().eval()
    payload = load_payload(args.features)
    features = payload['features']
    state = torch.zeros(features['n_tokens'], features['n_tokens'], dtype=torch.uint8)
    # predict() removes native coordinates, labels and observation masks before
    # calling the model, including when this is a validation feature payload.
    prediction, seconds = predict(model, features, state, True, payload['seed'], args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prediction, args.output)
    receipt = dict(checkpoint_sha256=sha(args.checkpoint), input_sha256=sha(args.features),
                   prediction_sha256=sha(args.output), seconds=seconds,
                   weight_variant=checkpoint['weight_variant'], step=checkpoint['step'],
                   seed=payload['seed'], samples=args.samples, cycles=1, diffusion_steps=200,
                   msa=True, contacts='all unknown')
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
