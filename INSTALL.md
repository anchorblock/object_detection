# Install Requiremnts

To setup your environments, run:


```bash
mkdir .venv
conda create -n venv python=3.9.0
conda activate venv
python3.9 -m venv ./.venv
source ./.venv/bin/activate
conda deactivate
conda deactivate
python3 -m pip install --upgrade pip
pip3 install --upgrade setuptools
pip3 install --no-cache-dir -r requirements.txt
```

or,

```bash
source ./.venv/bin/activate
```