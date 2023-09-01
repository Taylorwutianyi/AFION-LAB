#!/usr/bin/env sh
# To use this script, download git bash first
# may need to run in authorization mode

# link setting files
curpath=$(pwd -W)
if [ ! -f ~/.vimrc ]; then
    cmd.exe /C "mklink %UserProfile%\\.vimrc ${curpath////\\}\\.vimrc"
fi

if [ ! -f ~/.bashrc ]; then
    # mv -b ~/.bashrc ~/.bashrc.bk.$(date +%s)
    cmd.exe /C "mklink %UserProfile%\\.bashrc ${curpath////\\}\\.bashrc"
else
    echo 'already found ~/.bashrc, linking skipped'
fi

# install conda
if [ ! -d ~/miniconda3 ]; then
    echo 'Install Conda ...'
    curl https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe -o ~/Downloads/miniconda3-64.exe
    # curl https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe -o ~/Downloads/miniconda3-64.exe
    cmd.exe /C "%UserProfile%/Downloads/miniconda3-64.exe /S /D=%UserProfile%\\miniconda3"
    ~/miniconda3/Scripts/conda install -y conda=4.7.10
fi


# install pythonlab
source ~/miniconda3/etc/profile.d/conda.sh
if [[ $(conda env list | grep '^pythonlab\b') ]]; then
    echo 'pythonlab packages already Installed'
else
    echo 'Install pythonlab packages ...'
    conda env create -f pythonlab.yml
fi


# install NI
if [ ! -f ~/Downloads/NIVISA.zip ]; then
    echo 'Downloading NI ...'
    curl http://download.ni.com/support/softlib/visa/NI-VISA/18.0/Windows/NIVISA1800full.zip -o ~/Downloads/NIVISA.zip
    unzip ~/Downloads/NIVISA.zip -d ~/Downloads/NIVISA
fi

if [ -f ~/Downloads/NIVISA/setup.exe ]; then
    echo 'Installing NI ...'
    cmd.exe /C "%UserProfile%/Downloads/NIVISA/setup.exe /qf /AcceptLicenses yes /r:n /disableNotificationCheck"
    mv ~/Downloads/NIVISA/setup.exe ~/Downloads/NIVISA/setup_finished.exe
    echo 'Restart computer for National Instrument.'
fi

# set pythonlab in pythonpath
labpath=$(dirname "$(pwd -W)")
echo $labpath
SETX PYTHONPATH ${labpath////\\}


## backup scripts
# if [ ! -f ~/Downloads/NI4882.exe ]; then
#     echo 'Downloading NI ...'
#     curl http://download.ni.com/support/softlib/gpib/Windows/17.6/NI4882_1760f0.exe -o ~/Downloads/NI4882_1760f0.exe
#     unzip ~/Downloads/NI4882_1760f0.exe -d ~/Downloads/NI4882
# fi
#
# if [ -f ~/Downloads/NI4882/setup.exe ]; then
#     echo 'Installing NI ...'
#     cmd.exe /C "%UserProfile%/Downloads/NI4882/setup.exe /qf /AcceptLicenses yes /r:n /disableNotificationCheck"
# fi

# http://download.ni.com/support/softlib/visa/NI-VISA/18.0/Windows/NIVISA1800full.exe
# if [ ! -f ~/Downloads/NIVISA.exe ]; then
#     echo 'Downloading NI ...'
#     curl http://download.ni.com/support/softlib/visa/NI-VISA/18.0/Windows/NIVISA1800full.exe -o ~/Downloads/NIVISA.exe
#     unzip ~/Downloads/NIVISA.exe -d ~/Downloads/NIVISA
# fi

