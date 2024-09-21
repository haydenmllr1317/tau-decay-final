# tau-decay-ml
Summer 24

# Samples location

```
/isilon/data/users/mstamenk/tau-project-summer24/samples/preprocess15GeVMiniaodSampleWITHOUTpTCutIncl
```

# Installing CMSSW
Every time when opening a terminal (if not in the `.bashrc` file):

```
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

Once to install the package

```
cmsrel CMSSW_12_5_0
cd CMSSW_12_5_0/src/
cmsenv
```

Every time when coming back to work on the project

```
cd CMSSW_12_5_0/src/
cmsenv
```

# Downloading the package

```
cd CMSSW_12_5_0/src/
cmsenv
git clone git@github.com:mstamenk/tau-decay-ml.git
```
