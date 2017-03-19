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

 # Install SWIG >= 3 for box2d-py (See #83).
sudo apt-get -y install -t trusty-backports swig3.0
sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
pip install Box2D
```

Then also setup your pyCharm remote interpreter (conda remote) appropriately. 

To execute gym scripts, do:
```bash
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python random_agent.py
```
**Note**: this is working on Krypton but not Fermium. 
https://github.com/openai/gym/issues/366