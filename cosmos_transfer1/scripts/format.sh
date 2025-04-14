cosmos_root=$(git rev-parse --show-toplevel)
venv_folder=$cosmos_root/.venv
scripts_folder=$cosmos_root/cosmos_transfer1/scripts

echo "Formatting $cosmos_root"
if [ ! -d "$scripts_folder" ]; then
    echo "script has to be called from repo root dir!"
    exit -1
fi

if [ ! -d "$venv_folder" ]; then
    mkdir -p $venv_folder
    python3 -m pip install virtualenv
    python3 -m venv $venv_folder
fi

source $venv_folder/bin/activate

dependencies=($(pip freeze | grep -E 'pre-commit==3.7.1|flake8==7.1.0|black==24.4.2|isort==5.13.2|loguru|termcolor'))
if [ "${#dependencies[@]}" -ne 6 ]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install pre-commit==3.7.1
    python3 -m pip install flake8==7.1.0
    python3 -m pip install black==24.4.2
    python3 -m pip install isort==5.13.2
    python3 -m pip install loguru
    python3 -m pip install termcolor
fi
set -e
python3 $scripts_folder/ip_header.py
pre-commit install-hooks
pre-commit run --all