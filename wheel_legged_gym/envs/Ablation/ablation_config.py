from wheel_legged_gym.envs.wheel_legged_vmc_fyt.wheel_legged_vmc_fyt_config import (
    WheelLeggedVMCFYTCfg,
    WheelLeggedVMCFYTCfgPPO,
)


class FYTAblationNoSequenceCfg(WheelLeggedVMCFYTCfg):
    pass


class FYTAblationNoSequenceCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        policy_class_name = "ActorCritic"
        experiment_name = "ablation_fyt_no_sequence"
        run_name = "actor_critic"


class FYTAblationNoCurriculumCfg(WheelLeggedVMCFYTCfg):
    class terrain(WheelLeggedVMCFYTCfg.terrain):
        curriculum = False

    class commands(WheelLeggedVMCFYTCfg.commands):
        curriculum = False

        class ranges(WheelLeggedVMCFYTCfg.commands.ranges):
            lin_vel_x = [-2.5, 2.5]


class FYTAblationNoCurriculumCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        experiment_name = "ablation_fyt_no_curriculum"
        run_name = "no_terrain_no_command_curriculum"


class FYTAblationTerrainCurriculumOnlyCfg(WheelLeggedVMCFYTCfg):
    class commands(WheelLeggedVMCFYTCfg.commands):
        curriculum = False

        class ranges(WheelLeggedVMCFYTCfg.commands.ranges):
            lin_vel_x = [-2.5, 2.5]


class FYTAblationTerrainCurriculumOnlyCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        experiment_name = "ablation_fyt_terrain_curriculum_only"
        run_name = "terrain_curriculum_no_command_curriculum"


class FYTAblationNoEnhanceRewardsCfg(WheelLeggedVMCFYTCfg):
    class rewards(WheelLeggedVMCFYTCfg.rewards):
        class scales(WheelLeggedVMCFYTCfg.rewards.scales):
            tracking_lin_vel_enhance = 0.0
            base_height_enhance = 0.0


class FYTAblationNoEnhanceRewardsCfgPPO(WheelLeggedVMCFYTCfgPPO):
    class runner(WheelLeggedVMCFYTCfgPPO.runner):
        experiment_name = "ablation_fyt_no_enhance_rewards"
        run_name = "no_lin_vel_or_base_height_enhance"
