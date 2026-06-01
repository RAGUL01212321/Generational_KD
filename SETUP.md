This project includes helper files to set up a local Python environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To validate the environment, run:

```bash
python setup_local.py
```

This will check dependencies and attempt to download model artifacts listed in `model_configs/`.

Server / CI

```bash
# On your server (run from repo root)
./run_on_server.sh /opt/venvs/gen_kd   # creates venv and installs requirements
# or with tests:
RUN_TESTS=1 ./run_on_server.sh /opt/venvs/gen_kd
```

Docker

```bash
docker build -t generational_kd:latest .
docker run --rm -it generational_kd:latest
```
