# Setting up pyTorch and conda env

run the following

```bash
conda info --envs
conda create --name deep-learning
source activate deep-learning
conda install numpy matplotlib 
conda install pytorch torchvision cuda80 -c soumith
which python
pip install ipykernel
python -m ipykernel install --user --name deep-learning --display-name "deep-learning-python3"
```

Then also setup your pyCharm remote interpreter (conda remote) appropriately. 

To execute gym scripts, do:
```bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python random_agent.py
```

To run a juputer notebook with `xvfb`
```bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- jupyter notebook
```
**Note**: this is working on Krypton but not Fermium. 
https://github.com/openai/gym/issues/366