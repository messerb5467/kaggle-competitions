# Overview
My name is Brad Messer and I've come from a large applied mathematics background as well as engineering experience working on the IBM Z analytics strategy. This repo will contain all of the kaggle competitions I'm working through in my spare time. If you have any interest in discussing machine learning, AI or any other data science or mathematical related topics please keep in touch. I've done complexity theory, stochastic processes, markov chains, predator-prey systems, etc and would always enjoy a good talk in this area. I also enjoy taking classes on ai, machine learning, and anything else related to computational mathematics.

Eventually I'd like to work into the IBM AI Workflow certification as well as classes about gradient optimization, expectation maximization and the like. Thanks so much!

## Installing and using git lfs
Git LFS is installed and used within this repository to track larger files like the digit-recognizer/train.csv example. Depending on your environmental setup,
please follow this for [basic installation instructions](https://docs.github.com/en/free-pro-team@latest/github/managing-large-files/installing-git-large-file-storage) 
and this for [WSL installation](https://bigfont.ca/use-git-large-file-storage-lfs-in-the-windows-subsystem-for-linux-wsl/).
My personal environment runs inside WSL2 so there may be additional steps not linked to for git LFS basic installs. 
If that is the case, please let me know and I'll update the README.md file.

## Known issues in the conda environment
While installing from both anaconda and conda-forge, the following packages will bounce back and forth between two different versions:
  ca-certificates    conda-forge::ca-certificates-2020.12.~ --> anaconda::ca-certificates-2020.10.14-0
  certifi            conda-forge::certifi-2020.12.5-py38h5~ --> anaconda::certifi-2020.6.20-py38_0
  openssl            conda-forge::openssl-1.1.1i-h7f98852_0 --> anaconda::openssl-1.1.1h-h7b6447c_0
