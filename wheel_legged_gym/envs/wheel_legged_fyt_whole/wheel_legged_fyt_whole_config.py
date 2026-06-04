from wheel_legged_gym.envs.wheel_legged_vmc_fyt.wheel_legged_vmc_fyt_config import (
    WheelLeggedVMCFYTCfg,
    WheelLeggedVMCFYTCfgPPO,
)


class WheelLeggedFYTWholeCfg(WheelLeggedVMCFYTCfg):
    class asset(WheelLeggedVMCFYTCfg.asset):
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/wheel_legged_fyt_whole/urdf/wheel_legged_fyt_whole.urdf"
        name = "WheelLeggedFYTWhole"


class WheelLeggedFYTWholeCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        experiment_name = "wheel_legged_fyt_whole"
