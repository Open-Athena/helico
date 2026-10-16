"""Archive completed mmCIF predictions and official scoring evidence for release.

Extract archives into one directory. The aggregate prediction table's relative
paths then resolve across all stages. Original scorer cache records retain their
execution paths and identities for provenance.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

import pandas as pd


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def package(stage, output):
    stage, output = Path(stage).resolve(), Path(output)
    done = json.loads((stage / 'done.json').read_text())
    if done['stage'] != stage.name:
        raise ValueError('Stage identity mismatch')
    predictions = pd.read_csv(stage / 'predictions/prediction_reference.csv')
    if len(predictions) != done['structures']:
        raise ValueError('Prediction count differs from completion receipt')
    for row in predictions.itertuples():
        path = Path(row.prediction_path).resolve()
        path.relative_to(stage / 'predictions')
        if sha(path) != row.prediction_sha256:
            raise ValueError('Prediction content changed after scoring: ' + str(path))
    files = []
    for path in sorted(stage.rglob('*')):
        if not path.is_file():
            continue
        if path.is_symlink() or '.pending' in path.name:
            raise ValueError('Unexpected unfinished or linked scoring artifact')
        files.append(dict(path=str(path.relative_to(stage)),
                          bytes=path.stat().st_size, sha256=sha(path)))
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / (stage.name + '.tar.gz')
    pending = archive_path.with_suffix('.gz.pending')
    with pending.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|') as archive:
                for item in files:
                    info = tarfile.TarInfo(stage.name + '/' + item['path'])
                    info.size = item['bytes']; info.mode = 0o644; info.mtime = 0
                    with (stage / item['path']).open('rb') as handle:
                        archive.addfile(info, handle)
    pending.replace(archive_path)
    manifest = dict(stage=stage.name, completion=done, files=files,
                    archive=dict(path=archive_path.name,
                                 bytes=archive_path.stat().st_size,
                                 sha256=sha(archive_path)))
    marker = output / (stage.name + '.manifest.json')
    temporary = marker.with_suffix('.json.pending')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(marker)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = package(args.stage, args.output)
    print(json.dumps(dict(stage=result['stage'], archive=result['archive'])))
