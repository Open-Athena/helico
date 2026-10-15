"""Package completed, checksum-verified evaluation artifacts for Hugging Face.

Call ``package_stage(fs, uri, output)`` with an authenticated fsspec filesystem.
Credentials are supplied by the caller and are never included in the release.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def package_stage(fs, uri, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stage=uri.rstrip('/').rsplit('/',1)[-1]
    root=fs._strip_protocol(uri).rstrip('/')
    result=json.loads(fs.cat_file(uri+'/result.json'))
    if result['status']!='complete':raise ValueError('Evaluation is not complete')
    paths=[p for p in fs.find(uri) if '/diagnostics/' not in p and
           (p.endswith('.json') or p.endswith('.pt.gz'))]
    local=output/'.cache'/stage;local.mkdir(parents=True,exist_ok=True)
    def download(path):
        name=path.removeprefix(root+'/');dest=local/name
        if name==path or '..' in Path(name).parts:raise ValueError('Invalid artifact path')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():
            pending=dest.with_suffix(dest.suffix+'.pending')
            fs.get_file(path,str(pending));pending.replace(dest)
        return dict(path=name,bytes=dest.stat().st_size,sha256=sha(dest))
    with ThreadPoolExecutor(max_workers=8) as pool:
        files=list(pool.map(download,paths))
    by_name={r['path']:r for r in files};configs=set();sources=set();records=0
    ranks=[]
    for item in files:
        if not item['path'].endswith('.json'):continue
        record=json.loads((local/item['path']).read_text())
        if item['path'].startswith('ranks/'):ranks.append(record)
        if 'config_sha256' in record:configs.add(record['config_sha256'])
        if record.get('code_sha'):sources.add(record['code_sha'])
        if 'artifact' in record:
            artifact=by_name[record['artifact']]
            if artifact['sha256']!=record['artifact_sha256'] or artifact['bytes']!=record['artifact_bytes']:
                raise ValueError('Stored tensor artifact failed checksum verification')
            records+=1
    if len(ranks)!=result['ranks'] or any(r['status']!='complete' for r in ranks):
        raise ValueError('Incomplete evaluation rank records')
    if len(configs)!=1:raise ValueError('Mixed or absent immutable evaluation configurations')
    manifest=dict(stage=stage,source_uri=uri,result=result,config_sha256=next(iter(configs)),
                  source_commits=sorted(sources),tensor_records=records,files=files)
    path=output/(stage+'.tar');pending=path.with_suffix('.tar.pending')
    with tarfile.open(pending,'w') as archive:
        for item in sorted(files,key=lambda r:r['path']):
            source=local/item['path'];info=tarfile.TarInfo(stage+'/'+item['path'])
            info.size=source.stat().st_size;info.mode=0o644;info.mtime=0
            with source.open('rb') as handle:archive.addfile(info,handle)
    pending.replace(path)
    manifest['archive']=dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path))
    (output/(stage+'.manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
