# Linux / MacOS
$ brew install pyenv pyenv-virtualenv

$ export PYENV_PATH=$HOME/.pyenv
if which pyenv > /dev/null; then eval "$(pyenv init -)"; fi
if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi

$ mkdir streamlit && cd $_
$ pyenv virtualenv 3.10.17 streamlit
$ pyenv local streamlit

# streamlit 폴더로 이동하면 자동으로 (streamlit) 가상환경 모드가 적용됨

$ touch chat.py
$ streamlit run chat.py
$ ~/.pyenv/shims/streamlit run chat.py

# Windows
$ pyenv versions
$ cd streamlit
$ pyenv local 3.10.11