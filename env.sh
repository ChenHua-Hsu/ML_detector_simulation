#source /eos/user/t/tihsu/virtualenv/virtualenv3/bin/activate
#source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc8-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_102b/x86_64-centos7-gcc11-opt/setup.sh
centOSversion=$(awk -F= '/VERSION_ID/ {split($2, version, "."); print version[1]}' /etc/os-release | tr -d '"')
echo $centOSversion$
if [ "$centOSversion" -eq 9 ]; then
	#source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh 
	source /cvmfs/sft.cern.ch/lcg/views/LCG_104a_cuda/x86_64-el9-gcc11-opt/setup.sh 
else
	source /cvmfs/sft.cern.ch/lcg/views/LCG_102b_cuda/x86_64-centos7-gcc11-opt/setup.sh
fi

#source /cvmfs/sft.cern.ch/lcg/views/LCG_102b/x86_64-centos9-gcc11-opt/setup.sh
#source /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh
