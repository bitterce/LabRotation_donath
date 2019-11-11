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
```

## Installing Pycharm

[Click here](https://www.jetbrains.com/pycharm/download/download-thanks.html?platform=mac&code=PCC) to download Pycharm.

To see if everything works as expected, we try to run a small test code with Pycharm.
Go to your applications and open Pycharm. 
**This step is important, as we need to shift the working directory from our local machine to the TRANSFER server**. To do this, press 'open' and navigate via the transfer to the NN folder, as shown in the pictures: 
<img width="775" alt="open" src="https://user-images.githubusercontent.com/48200405/68580283-54962700-0476-11ea-9ecf-f78ff60952fb.png">
<img width="824" alt="open2" src="https://user-images.githubusercontent.com/48200405/68580670-2107cc80-0477-11ea-959b-2fc0259a0a4e.png">
