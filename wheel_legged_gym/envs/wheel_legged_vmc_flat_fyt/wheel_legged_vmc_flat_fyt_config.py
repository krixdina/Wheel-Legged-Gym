from wheel_legged_gym.envs.wheel_legged_vmc_fyt.wheel_legged_vmc_fyt_config import (
    WheelLeggedVMCFYTCfg,
    WheelLeggedVMCFYTCfgPPO,
)


class WheelLeggedVMCFlatFYTCfg(WheelLeggedVMCFYTCfg):

    class terrain(WheelLeggedVMCFYTCfg.terrain):
        mesh_type = "plane"


class WheelLeggedVMCFlatFYTCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc_flat_fyt"
        max_iterations = 2000
