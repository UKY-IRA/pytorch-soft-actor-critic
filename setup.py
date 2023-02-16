from distutils.core import setup

setup(name='uav_sac',
      version='1.0',
      description='An SAC training algorithm for trained UAV exploration of gas plumes',
      author="Josh Ashley",
      author_email="jashley2017@gmail.com",
      packages=["uav_sac", "uav_sac.environments", "uav_sac.networks", "uav_sac.scripts"]
)
'''
py_modules=[
      *["uav_sac." + module for module in ["main", "sac", "replay_memory", "gen_map", "verify"]],  # main module
      *["uav_sac.environments." + module for module in ["belief_model", "uav_explorer" "belief2d", "simple2duav"]],  # environment submodule
      *["uav_sac.networks." + module for module in ["conv2d_model"]],  # networks submodule
      *["uav_sac.scripts." + module for module in ["generate_from_pompy", "display_json", "make_satmap"]]  # scripts submodule
]
'''