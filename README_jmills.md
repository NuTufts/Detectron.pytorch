# An Adjustment to Detectron to use LArCV particle physics data as input images


Getting Started:


First clone the repository:
git clone https://github.com/NuTufts/Detectron.pytorch.git

In order to use LArCV1 format, with the latest updates, swap to branch:

larcv1_mcc9

git checkout larcvv1_mcc9

Make sure you are tracking from remote, not on just a local branch.

Next you need to build the repository. You'll need
-pytorch (tested on 1.0.1.post2)
-with CUDA 10

and other smaller dependencies detailed in the other README

To Build:
cd lib/
source make.sh

The output is a little hard to parse, so check for errors!

In order to train code use python script
python tools/train_particle.py (see file for list of args)

In order to test code use python script
python tools/infer_particle.py (see file for list of args)

You'll need to have sourced larcv for these to work. You'll also need a cfg file
look for those in configs/baselines/mills_config_#.yaml where I use # to
specify the u,v,y planes.

More default config params live in lib/core/config.py
But the yaml overwites any params living in both files at script start.
