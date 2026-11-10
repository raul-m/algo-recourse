"""Run an explicit, disjoint list of coefficient jobs with the existing trainer."""
import argparse
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',choices=['credit','oulad'],required=True)
parser.add_argument('--setting',choices=['c1','c2'],required=True)
parser.add_argument('--alphas',type=float,nargs='+',required=True)
args=parser.parse_args()
assert len(args.alphas)==len(set(args.alphas))
for alpha in args.alphas:
    subprocess.run([sys.executable,str(Path(__file__).with_name('cf_feasibility_multidata.py')),
        'train','--dataset',args.dataset,'--setting',args.setting,'--alpha',str(alpha)],check=True)
