"""Sprint level: a planar humanoid runs as far as it can in 10 seconds.

Modeled on mujoco_playground/_src/dm_control_suite/walker.py (playground 0.2.0).
The reward is a weighted sum of named terms; every weight comes from the config
so the future app can toggle and weight each term.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "humanoid2d.xml"

# Named reward terms, in a fixed order. Each one's weight is in config.reward_weights.
REWARD_TERMS = (
    "forward_velocity", "alive", "upright", "control_cost",
    "alternation", "foot_slip", "symmetry",
)

# Limb joints in qpos/qvel/actuator order (the three root joints come first).
LIMB_JOINTS = (
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_shoulder", "right_elbow",
    "left_shoulder", "left_elbow",
)
FEET = ("right", "left")


def quiet_warp_overflow(mjx_model: mjx.Model) -> mjx.Model:
  """Turn off Warp's solver/linesearch iteration-limit warnings.

  The solver deliberately runs a small fixed number of iterations. With
  mujoco-mjx 3.14, Warp prints a warning from the GPU every time that limit is
  hit, which floods the log and made WalkerRun steps ~6x slower on the RTX 3070
  (46k -> 289k steps/s once disabled). Same fix as MuJoCo Playground main
  (commit fb246cd), which is not in the 0.2.0 release.
  """
  if mjx_model.impl != mjx.Impl.WARP:
    return mjx_model
  from mujoco.mjx.third_party import mujoco_warp as mjw  # pylint: disable=g-import-not-at-top

  mask = ~(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
  warn = int(mjx_model.opt._impl.warn_overflow)  # pylint: disable=protected-access
  return mjx_model.tree_replace({"opt._impl.warn_overflow": warn & mask})


def put_model(mj_model: mujoco.MjModel, impl: str) -> mjx.Model:
  return quiet_warp_overflow(mjx.put_model(mj_model, impl=impl))


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.025,
      sim_dt=0.0025,
      episode_length=400,  # 10 s of sim time at ctrl_dt
      action_repeat=1,
      impl="warp",
      # Warp buffers. naconmax is the contact budget summed over ALL parallel
      # worlds; sprinter.train sizes it from num_envs. njmax is per world.
      naconmax=2048 * 16,
      njmax=80,
      # Reset: standing keyframe plus uniform noise of this half-width.
      reset_noise_qpos=0.05,  # rad, limb joints
      reset_noise_qvel=0.1,  # rad/s and m/s, all joints
      # Termination.
      min_torso_height=0.65,  # m, height of the hip joints (torso frame)
      max_torso_pitch=1.0,  # rad, forward or backward lean
      # Reward.
      max_forward_velocity=12.0,  # m/s, clip for the forward_velocity term
      foot_contact_threshold=0.005,  # m, sole height that counts as contact
      # A landing counts as a step only after this much air time (no tap-dancing),
      # and earns the full alternation bonus after full_step_air_time.
      min_step_air_time=0.1,  # s
      full_step_air_time=0.25,  # s
      # Time constant of each foot's running-average ground contact (symmetry
      # term). Long enough to smooth out the ripple within each stride.
      contact_average_time=2.0,  # s
      reward_weights=config_dict.create(
          forward_velocity=1.0,
          alive=1.0,
          upright=0.0,
          control_cost=1e-3,
          alternation=0.0,
          foot_slip=0.0,
          symmetry=0.0,
      ),
  )


class Sprint(mjx_env.MjxEnv):
  """Run forward as far as possible in a fixed time without falling."""

  def __init__(
      self,
      config: Optional[config_dict.ConfigDict] = None,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config or default_config(), config_overrides)
    self._xml_path = XML_PATH.as_posix()
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = put_model(self._mj_model, self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    m = self._mj_model
    self._torso_id = m.body("torso").id
    self._limb_qpos = mjx_env.get_qpos_ids(m, LIMB_JOINTS)
    self._limb_qvel = mjx_env.get_qvel_ids(m, LIMB_JOINTS)
    assert list(self._limb_qpos) == list(range(3, 13)), "root joints must come first"
    self._rootx, self._rootz, self._rooty = 0, 1, 2
    self._foot_sites = jp.array([
        [m.site(f"{side}_heel").id, m.site(f"{side}_toe").id] for side in FEET
    ])
    self._foot_radius = float(m.geom_size[m.geom("right_foot").id][0])
    self._init_qpos = jp.array(m.keyframe("stand").qpos)
    w = self._config.reward_weights
    self._weights = {k: float(w[k]) for k in REWARD_TERMS}

  # ---------------------------------------------------------------------------
  # Reset / step

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_q, k_v = jax.random.split(rng, 3)
    cfg = self._config
    qpos = self._init_qpos.at[self._limb_qpos].add(
        jax.random.uniform(
            k_q, (len(LIMB_JOINTS),),
            minval=-cfg.reset_noise_qpos, maxval=cfg.reset_noise_qpos,
        )
    )
    qvel = jax.random.uniform(
        k_v, (self.mjx_model.nv,),
        minval=-cfg.reset_noise_qvel, maxval=cfg.reset_noise_qvel,
    )
    data = self._make_data(qpos, qvel)
    # Joint noise can push a sole into the floor or lift it: re-seat the lower
    # foot 0.5 mm above the floor so every episode starts at rest on the ground.
    sole = self._sole_heights(data)
    qpos = qpos.at[self._rootz].add(0.0005 - jp.min(sole))
    data = self._make_data(qpos, qvel)

    metrics = {f"reward/{k}": jp.zeros(()) for k in REWARD_TERMS}
    metrics.update(
        distance_x=jp.zeros(()), speed_per_step=jp.zeros(()), fell=jp.zeros(())
    )
    obs = self._get_obs(data)
    reward, done = jp.zeros(2)
    info = {
        "rng": rng,
        "last_foot": jp.array(-1, dtype=jp.int32),  # foot of the last step: 0 right, 1 left
        "air_time": jp.zeros(2),  # seconds each foot has been off the floor
        "contact_avg": jp.full(2, 0.5),  # running-average ground contact of each foot
    }
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    cfg = self._config
    # Playground's auto-reset restores data (time = 0) but not info: start the
    # gait bookkeeping over at the first step of every episode.
    first = state.data.time < 0.5 * self.sim_dt
    last_foot = jp.where(first, -1, state.info["last_foot"])
    air = jp.where(first, 0.0, state.info["air_time"])
    contact_avg = jp.where(first, 0.5, state.info["contact_avg"])
    was_down = self.foot_contacts(state.data) > 0.5

    data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
    down = self.foot_contacts(data) > 0.5
    alternation, last_foot, air = self._alternation(was_down, down, air, last_foot)
    slip = self._foot_slip(state.data, data, down)
    imbalance, contact_avg = self._contact_imbalance(contact_avg, down)

    dx = data.qpos[self._rootx] - state.data.qpos[self._rootx]
    velocity = dx / self.dt
    pitch = data.qpos[self._rooty]
    torso_height = data.xpos[self._torso_id, 2]

    nan = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    fell = (
        (torso_height < cfg.min_torso_height)
        | (jp.abs(pitch) > cfg.max_torso_pitch)
        | nan
    )
    done = fell.astype(jp.float32)

    terms = {
        "forward_velocity": jp.clip(
            velocity, -cfg.max_forward_velocity, cfg.max_forward_velocity
        ),
        "alive": 1.0 - done,
        "upright": jp.cos(pitch),
        "control_cost": -jp.sum(jp.square(action)),
        "alternation": alternation,
        "foot_slip": -slip,
        "symmetry": -imbalance,
    }
    rewards = {k: self._weights[k] * terms[k] for k in REWARD_TERMS}
    reward = sum(rewards.values())

    obs = self._get_obs(data)
    # Never let a NaN reach the learner; the auto-reset wrapper restarts the env.
    reward = jp.where(nan, 0.0, reward)
    obs = jp.where(jp.isnan(obs), 0.0, obs)
    dx = jp.where(nan, 0.0, dx)

    # Start from the incoming dict: Brax's eval wrapper adds its own keys to it.
    metrics = dict(state.metrics)
    metrics.update({f"reward/{k}": jp.where(nan, 0.0, v) for k, v in rewards.items()})
    metrics.update(
        distance_x=dx,
        speed_per_step=dx / self.dt,  # Brax divides *_per_step metrics by episode length
        fell=done,
    )
    info = {**state.info, "last_foot": last_foot, "air_time": air, "contact_avg": contact_avg}
    return state.replace(
        data=data, obs=obs, reward=reward, done=done, metrics=metrics, info=info
    )

  def _alternation(self, was_down, down, air, last_foot):
    """+1 for landing on the other foot than last time, -1 for the same foot.

    A landing only counts after min_step_air_time in the air; the bonus is
    scaled by air time so tiny shuffle steps earn less. Both feet landing
    together (a two-footed hop) costs -1 and clears the memory.
    """
    cfg = self._config
    landed = down & ~was_down & (air >= cfg.min_step_air_time)
    both = landed[0] & landed[1]
    one = landed[0] ^ landed[1]
    foot = jp.where(landed[0], 0, 1).astype(jp.int32)
    bonus = jp.minimum(air[foot] / cfg.full_step_air_time, 1.0)
    value = jp.where(last_foot == -1, 0.0, jp.where(last_foot == foot, -1.0, bonus))
    value = jp.where(both, -1.0, jp.where(one, value, 0.0))
    last_foot = jp.where(both, -1, jp.where(one, foot, last_foot)).astype(jp.int32)
    air = jp.where(down, 0.0, air + self.dt)
    return value, last_foot, air

  def _contact_imbalance(self, contact_avg: jax.Array, down: jax.Array):
    """|right - left| of each foot's running-average ground contact.

    A symmetric gait keeps both averages level; a limp (one foot planted far
    longer than the other) pulls them apart.
    """
    rate = self.dt / self._config.contact_average_time
    contact_avg = contact_avg + rate * (down.astype(jp.float32) - contact_avg)
    return jp.abs(contact_avg[0] - contact_avg[1]), contact_avg

  def _foot_slip(self, data0: mjx.Data, data1: mjx.Data, down: jax.Array) -> jax.Array:
    """Squared horizontal speed (m^2/s^2) of each grounded foot's lowest point, summed.

    A planted or rolling foot pivots about its lowest point, which stays put;
    a foot dragged or skated along the floor moves. Squaring keeps the small
    slips of a landing cheap and makes a dragged toe (~6 m/s) expensive.
    """
    s0 = data0.site_xpos[self._foot_sites]  # (2 feet, heel/toe, xyz)
    s1 = data1.site_xpos[self._foot_sites]
    low = jp.argmin(s1[..., 2], axis=-1)  # lower site of each foot
    dx = jp.take_along_axis(s1[..., 0] - s0[..., 0], low[:, None], axis=-1)[:, 0]
    return jp.sum(jp.where(down, jp.square(dx / self.dt), 0.0))

  # ---------------------------------------------------------------------------
  # Helpers

  def _make_data(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    return mjx.forward(self.mjx_model, data)

  def _sole_heights(self, data: mjx.Data) -> jax.Array:
    """Lowest point of each foot capsule, shape (2,) in FEET order."""
    site_z = data.site_xpos[self._foot_sites, 2]  # (2 feet, heel/toe)
    return jp.min(site_z, axis=-1) - self._foot_radius

  def foot_contacts(self, data: mjx.Data) -> jax.Array:
    """1.0 where a foot is on (or within the threshold of) the floor."""
    return (
        self._sole_heights(data) < self._config.foot_contact_threshold
    ).astype(jp.float32)

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    pitch = data.qpos[self._rooty]
    return jp.concatenate([
        data.qpos[self._limb_qpos],  # 10 joint angles
        data.qvel[self._limb_qvel],  # 10 joint velocities
        data.xpos[self._torso_id, 2:3],  # torso (hip) height
        jp.sin(pitch)[None],
        jp.cos(pitch)[None],
        data.qvel[:3],  # root x/z velocity and pitch rate
        self.foot_contacts(data),  # right, left
    ])

  # ---------------------------------------------------------------------------
  # MjxEnv interface

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model

  @property
  def reward_weights(self) -> Dict[str, float]:
    return dict(self._weights)
