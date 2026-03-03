# Test data for TestFoldRealProtein

Offline data for folding 1MBN (sperm whale myoglobin). No internet required.

| File        | Source                         | Description              |
|-------------|--------------------------------|--------------------------|
| `1mbn.cif.gz` | RCSB PDB (files.rcsb.org)   | Ground truth structure   |
| `1mbn.a3m`    | ColabFold MMseqs2 API         | MSA in A3M format        |

To regenerate (requires internet):

```bash
# CIF
curl -sL -o tests/data/1mbn.cif.gz https://files.rcsb.org/download/1MBN.cif.gz

# A3M
PYTHONPATH=src python -c "
from pathlib import Path
from helico.msa_server import run_mmseqs2
MBN = 'VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG'
a3m = run_mmseqs2(MBN, result_dir='tests/data/1mbn_A')[0]
Path('tests/data/1mbn.a3m').write_text(a3m)
"
rm -rf tests/data/1mbn_A_env  # optional: remove cache dir
```
