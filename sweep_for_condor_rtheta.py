import os, sys
import optparse, argparse
import yaml
import wandb

# Usage: python sweep_for_condor.py --python_cfg trans_tdsm.py --config_file configs/... --afs_dir ... --n_run xxx

CWD = os.getcwd()
def prepare_shell(shell_file, command, condor, FarmDir, afs_dir):

  with open(os.path.join(FarmDir, shell_file), 'w') as shell:
    shell.write('#!/bin/bash\n')
    shell.write('WORKDIR={}\n'.format(afs_dir))
    shell.write('cd ${WORKDIR}\n')
    shell.write('source script/env.sh\n')
    shell.write('cd {}\n'.format(CWD))
    shell.write(command)

  condor.write('cfgFile=%s\n'%shell_file)
  condor.write('queue 1\n')

if __name__ == '__main__':
  usage = 'usage: %prog[options]'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('--config_file', type=str, default = '/eos/user/c/chenhua/copy_tdsm_encoder_sweep16/configs/rtheta_1033.yml',help = 'configuration file for wandb')
  parser.add_argument('--python_cfg', type=str, default = 'trans_tdsm_rtheta.py', help = 'python file to run the code')
  parser.add_argument('--n_run', type=int, default = 5, help = 'number of runs')
  parser.add_argument('--dryRun', action='store_true', help = 'not submit to condor')
  parser.add_argument('--afs_dir', type=str, default = '/afs/cern.ch/user/c/chenhua', help='workspace in afs space')
  parser.add_argument('--JobFlavour', type=str, default = 'tomorrow', help='JobFlavour for condor')
  args = parser.parse_args()

  # Read configuration file for sweep.
  with open(args.config_file, 'r') as file:
    sweep_yml = yaml.safe_load(file)

  # Define extra information on top of configuration file for running purpose i.e. python_cfg argument.
  CWD = os.getcwd()
  sweep_yml['program'] = os.path.join(CWD, args.python_cfg)
  sweep_yml['parameters']['work_dir'] = {'value': CWD}
  sweep_yml['parameters']['switches'] = {'value': '1111'}
  sweep_yml['parameters']['condor']   = {'value': 1}
  sweep_yml['parameters']['inputs']= {'value': os.path.join(CWD, "datasets_new")}
  sweep_yml['parameters']['preprocessor'] = {'value': os.path.join(CWD, "datasets_new/dataset_2_padded_rtheta_new_transformation_nentry1033To1161.pt")}


  # Create necessary work space in afs space (condor can not be submitted from eos space)
  afs_dir = args.afs_dir
  farm_dir = os.path.join(afs_dir, 'Farm')
  os.system('mkdir -p {}'.format(farm_dir))

  script_dir = os.path.join(afs_dir, 'script')
  os.system('mkdir -p {}'.format(script_dir))
  os.system('cp env.sh {}/env.sh'.format(script_dir))

  home_dir = os.path.expanduser("~")


  os.system('cp -r util {}/.'.format(afs_dir))

  condor = open(os.path.join(farm_dir, 'condor.sub'), 'w')
  condor.write('output = %s/job_common_$(Process).out\n'%farm_dir)
  condor.write('error  = %s/job_common_$(Process).err\n'%farm_dir)
  condor.write('log    = %s/job_common_$(Process).log\n'%farm_dir)
  condor.write('executable = %s/$(cfgFile)\n'%farm_dir)
  condor.write('request_GPUs = 1\n')
  condor.write('+JobFlavour = "%s"\n'%args.JobFlavour)

  # Create sweep project
  project_name = args.config_file.split('.')[0].split('_',1)[1].replace("/", "_")#args.config_file.split('.')[0].split('_',1)[1]
  sweep_id = wandb.sweep(sweep_yml, project="NCSM-"+project_name)
  for run in range(args.n_run):
    # Each shell file will launch a sweep using shell command, wandb would handle the workflow
    shell_file = 'sweep_run{}.sh'.format(run)
    command = '{}/.local/bin/wandb agent calo_tNCSM/{}/{} --count {}'.format(home_dir, "NCSM-"+project_name, sweep_id, "1")
    prepare_shell(shell_file, command, condor, farm_dir, afs_dir)
  condor.close()

  # dryRun: not submit to condor
  if not args.dryRun:
    os.system('cd {};condor_submit {}'.format(afs_dir, os.path.join(farm_dir, 'condor.sub')))
