#!/bin/sh
brew update
brew install pyenv
sudo chmod 777 /Users/hagi/.bash_profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
# pyenv install 3.9.0
conda create --name py39 python==3.9.1

# To activate this environment, use
#
#     $ conda activate py39
#
# To deactivate an active environment, use
#
#     $ conda deactivate
