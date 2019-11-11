# Preparations and setups
## Installing python 3.7

Open your terminal by **command + space** + terminal and check for the python version by:
```
python --version
```
You'll most likely see something like this
```
out: Python 2.7.16
```
We need to upgrade this to python 3.7 by following these steps consecutively
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

export PATH="/usr/local/opt/python/libexec/bin:/usr/local/bin:$PATH"

brew install python

pip install virtualenv

virtualenv pytorch

source pytorch/bin/activate
```

If you've successfully activate your environment you should see it in your terminal like this:
```
(pytorch)
