from distutils.core import setup

setup(name='uav_sac',
      version='1.0',
      description='An SAC training algorithm for trained UAV exploration of gas plumes',
      author="Josh Ashley",
      author_email="jashley2017@gmail.com",
      namespace_packages=["uav_sac", "uav_sac.environments", "uav_sac.networks", "uav_sac.scripts"]
)