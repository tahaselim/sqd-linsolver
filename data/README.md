# Hardware data

Per-time counts of the IBM Quantum jobs used in the paper, written once by cell B15 of
`sqd_linear_systems_selfcontained.ipynb` (needs `IBM_CLOUD_TOKEN`; no new job is submitted):

* `ibm_fez_4x4_counts.json` — job `dbh4djsdgr2s73csnj3g` (Fig. 3, Table III)
* `ibm_fez_banded_scaling_counts.json` — jobs `dbeco3fgdj9s73flhr50`, `dbeco6fsstms73fbqrq0`, `dbecoengdj9s73flhrq0`

With these files present, both notebooks reproduce the hardware results without an IBM Quantum account.
`reproduce_paper.ipynb` finds the 4x4 file here (`../data/`) or in its own `data/` folder.
