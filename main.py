#Python imports
import os
import shutil
import sys
import logging

#From Acme
from absl import app
from acme.jax import experiments
from acme.utils import lp_utils
import launchpad as lp
import launchpad.context
from launchpad.nodes.python.local_multi_processing import PythonProcess

#Local files
import helpers
import gpu
import ppo
from ppo import experimentConfig as configClass


seed = 1

# Parse command line arguments, or use defaults.
# Useful for running batches of experiments from shell scripts
try:
    numSteps = int(float(sys.argv[1]))
except:
    numSteps = int(1e10)
assert isinstance(numSteps, int)
evalEvery = 500_0000
try:
    trainFresh = int(sys.argv[5]) == 1
except:
    trainFresh = False
    print('Train fresh fail to parse')
try:
    runDistributed = int(sys.argv[7]) == 1
except:
    runDistributed = False

try:
    useGPU = int(sys.argv[6]) == 1
except:
    useGPU = True
if useGPU:
    gpuNum = 0
else:
    gpuNum = -1


def build_PPO_config(gymEnv, obsNum, batchSize):
    """Builds PPO experiment config which can be executed in different ways."""
    # Create an environment, grab the spec, and use it to create networks.
    days = 240
    numDaysToObserve = 2
    obss = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    if isinstance(obsNum, int):
        obs = obss[0:obsNum+1]
    else:
        obs = obsNum
    tensorboardDir = 'coach_' + gymEnv + '_expt' + '{}_obsNum'.format(obsNum)

    config = ppo.PPOConfig(
        normalize_advantage=True,
        normalize_value=True,
        obs_normalization_fns_factory=ppo.build_mean_std_normalizer,
        unroll_length=8,
        discount=0.99, ppo_clipping_epsilon=0.1, entropy_cost=1e-5,
        batch_size=batchSize,
        learning_rate=3e-4)
    ppo_builder = ppo.PPOBuilder(config)

    def make_logger(label, steps_key='learner_steps', i=0):
        # gpu.SetGPU(-1)
        from acme.utils.loggers.terminal import TerminalLogger
        from acme.utils.loggers.tf_summary import TFSummaryLogger
        from acme.utils.loggers import base, aggregators
        summaryDir = "tensorboardOutput/" + tensorboardDir
        terminal_logger = TerminalLogger(label=label, print_fn=logging.info)
        tb_logger = TFSummaryLogger(summaryDir, label=label, steps_key=steps_key)
        serialize_fn = base.to_numpy
        logger = aggregators.Dispatcher([tb_logger, terminal_logger], serialize_fn)
        return logger

    checkpointingConfig = configClass.CheckpointingConfig()
    checkpointingConfig.max_to_keep = 5
    checkpointingConfig.directory = '/home/kenny/acme/coach/' + 'sim' + '/PPO/{}_obsnum'.format(obs[-1])
    checkpointingConfig.time_delta_minutes = 10
    checkpointingConfig.add_uid = False

    env_factory = lambda _: helpers.make_coach_env(gymEnv, days, numDaysToObserve, obs)

    layer_sizes = (512, 512)
    return experiments.ExperimentConfig(
        builder=ppo_builder,
        environment_factory=env_factory,
        network_factory=lambda spec: ppo.make_networks(spec, layer_sizes),
        seed=seed,
        max_num_actor_steps=int(numSteps),
        logger_factory=make_logger,
        checkpointing=checkpointingConfig,
        evaluator_factories=[]
    )


def launchDistributed(experiment_config, numActors=1, numLearners=1):
    terminals = ['gnome-terminal', 'gnome-terminal-tabs', 'xterm',
                 'tmux_session', 'current_terminal', 'output_to_files']
    # For a distributed run, you have many options for the program std output.
    # xterm and tmux require additional installs.
    # gnome-terminal-tabs is to me the most straightforward/organized

    print("______________________________________________")
    print("numactors: {}, numlearners: {}".format(numActors, numLearners))
    print("______________________________________________")

    program = experiments.make_distributed_experiment(
        experiment=experiment_config,
        num_actors=numActors, num_learner_nodes=numLearners)

    # In Acme/Launchpad: how to set per-process environment variables for gpu memory allocation.

    # E.g., the following block will do just that, by passing in a dictionary of env variables to the Launchpad "PythonProcess" class.
    # Launchpad will then take this dictionary describing all resources available to each process type, and set about launching each with lp.launch.
    # 'learner':
    # PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1',
    #                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
    #                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true'))



    resources = {
        # The 'actor' and 'evaluator' keys refer to
        # the Launchpad resource groups created by Program.group()
        'actor':
            PythonProcess(  # Dataclass used to specify env vars and args (flags) passed to the Python process
                env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                         XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'evaluator':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'counter':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'replay':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform')),
        'learner':
            PythonProcess(env=dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
                                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='platform',
                                   XLA_PYTHON_CLIENT_MEM_FRACTION='.40', TF_FORCE_GPU_ALLOW_GROWTH='true')),
    }
    #Using the defined distributed experiment config "program" process-group-specific resources "resources", launch the distributed training run

    #For launch_type, I recommend local_multi_processing. If you use threading, you don't get the benefit of splitting worker stdout into multiple tabs.
    #That said, if in general you are using licensed software in your environment, multiple processes may not be allowed; in this case local_multi_threading will also work.
    #However, in my limited experience with Acme, processes rather than threads seem to give better performance and utilization of hardware.
    worker = lp.launch(program, xm_resources=lp_utils.make_xm_docker_resources(program),
                       launch_type=launchpad.context.LaunchType.LOCAL_MULTI_PROCESSING,
                       terminal=terminals[1], local_resources=resources)
    worker.wait()


def main(_):
    username = 'YOUR_NAME'
    saveDir = '/home/{}/acme/coach/'.format(username)

    envs = ['sim', 'irl']
    envNum = 0 #Choose between training in simulation using HabitEnvSim,
    # or training in real life with the google sheet.
    envName = envs[envNum]
    if envNum == 0:
        numActors = 32 #Change this for your number of threads
    else:
        numActors = 1
    obs = [0,1,2,3,4,5,6,7,8,9] # Change this for the subset of the habits
    # you would like to train (really only applicable to simulation)
    batch_size = int(2 ** 8) # 256 seems to work well
    
    if trainFresh: #delete checkpoints
        dirToRm = saveDir + envName + '/PPO/{}_obsnum'.format(obs[-1])
        if os.path.isdir(dirToRm):
            shutil.rmtree(dirToRm)
    # Trying PPO...
    experiment_config = build_PPO_config(envName, obs, batch_size)
    if runDistributed:
        launchDistributed(experiment_config=experiment_config, numActors=numActors, numLearners=1)
    else:
        experiments.run_experiment(experiment_config, eval_every=evalEvery)


if __name__ == '__main__':
    gpu.SetGPU(1, runDistributed)
    app.run(main)
