python -c "
import isaacgym
import torch
from hydra import compose, initialize_config_dir
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import os
cfg_dir = os.path.join(os.path.dirname(isaacgymenvs.__file__), 'cfg')
with initialize_config_dir(config_dir=cfg_dir):
    cfg = compose(config_name='config', overrides=[
        'task=TacSLTaskBulb', 'train=TacSLTaskInsertionPPO_LSTM_dict_AAC',
        'num_envs=1', 'headless=True',
    ])
    task_cfg = omegaconf_to_dict(cfg.task)
    env = isaacgym_task_map['TacSLTaskInsertion'](
        cfg=task_cfg, rl_device='cuda:0', sim_device='cuda:0',
        graphics_device_id=0, headless=True,
        virtual_screen_capture=False, force_render=True,
    )
obs = env.reset()
if isinstance(obs, dict) and 'obs' in obs:
    obs = obs['obs']
print('=== obs type:', type(obs))
if isinstance(obs, dict):
    for k, v in obs.items():
        t = torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v
        print(f'  {k}: shape={tuple(t.shape)} dtype={t.dtype}')
else:
    print(f'  shape={tuple(obs.shape)} dtype={obs.dtype}')
# Check for camera-related attributes on the env
for attr in ['camera_obs', 'camera_tensors', 'wrist_obs', 'image_obs',
             'obs_dict', 'camera_sensor', 'sensors', 'wrist_tensor']:
    if hasattr(env, attr):
        val = getattr(env, attr)
        print(f'env.{attr} = {type(val)}',
              {k: type(v) for k,v in val.items()} if isinstance(val, dict) else '')
"
