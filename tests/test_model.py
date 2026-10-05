"""Checks on the humanoid MJCF: loads, sane proportions, stable physics."""

import itertools

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from sprinter.envs import sprint

XML = sprint.XML_PATH.as_posix()
LIMB_HINGES = sprint.LIMB_JOINTS


@pytest.fixture(scope="module")
def model():
  return mujoco.MjModel.from_xml_path(XML)


def _capsule_min_z(m, d, geom):
  g = m.geom(geom).id
  axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
  half = m.geom_size[g][1]
  c = d.geom_xpos[g]
  return min(c[2] + half * axis[2], c[2] - half * axis[2]) - m.geom_size[g][0]


def test_model_loads_with_expected_structure(model):
  m = model
  assert m.nu == 10
  assert [m.joint(i).name for i in range(3)] == ["rootx", "rootz", "rooty"]
  assert m.jnt_type[0] == m.jnt_type[1] == mujoco.mjtJoint.mjJNT_SLIDE
  assert np.allclose(m.jnt_axis[0], [1, 0, 0]) and np.allclose(m.jnt_axis[1], [0, 0, 1])
  for name in ("rooty",) + LIMB_HINGES:
    j = m.joint(name).id
    assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    assert np.allclose(m.jnt_axis[j], [0, 1, 0]), name
  for name in LIMB_HINGES:
    assert m.actuator(name).trntype == mujoco.mjtTrn.mjTRN_JOINT
    assert np.allclose(m.actuator_ctrlrange[m.actuator(name).id], [-1, 1])
  for side in ("left", "right"):
    assert m.body(f"{side}_hand").id > 0
    for site in ("heel", "toe"):
      assert m.site(f"{side}_{site}").id >= 0
    assert m.site(f"{side}_hand_grip").id >= 0


def test_human_proportions(model):
  m = model
  assert 65 < m.body_subtreemass[m.body("torso").id] < 75
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  head = m.geom("head").id
  height = d.geom_xpos[head][2] + m.geom_size[head][0]
  assert 1.7 <= height <= 1.8
  # Knees and elbows bend one way only.
  rng = {n: np.rad2deg(m.jnt_range[m.joint(n).id]) for n in LIMB_HINGES}
  for side in ("left", "right"):
    lo, hi = rng[f"{side}_knee"]
    assert lo >= 0 and 120 <= hi <= 160
    lo, hi = rng[f"{side}_elbow"]
    assert hi <= 0 and -160 <= lo <= -120


def test_body_parts_only_collide_with_floor(model):
  m = model
  floor = m.geom("floor").id
  body_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]

  def can_collide(a, b):
    return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))

  for a, b in itertools.combinations(body_geoms, 2):
    assert not can_collide(a, b), (m.geom(a).name, m.geom(b).name)
  colliding = {m.geom(g).name for g in body_geoms if can_collide(g, floor)}
  for side in ("left", "right"):
    assert {f"{side}_foot", f"{side}_shin", f"{side}_hand"} <= colliding
  assert len(colliding) <= 8  # keep the contact count low


def test_passive_drop_is_realistic():
  m = mujoco.MjModel.from_xml_path(XML)  # own copy: this test changes opt flags
  m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.keyframe("stand").id)
  d.qpos[1] += 0.5  # start 0.5 m above the floor
  mujoco.mj_forward(m, d)
  com0 = d.subtree_com[1][2]
  energy0 = d.energy.sum()
  max_energy, max_speed = energy0, 0.0
  steps = int(3.0 / m.opt.timestep)
  for i in range(steps):
    mujoco.mj_step(m, d)
    if i + 1 == int(0.2 / m.opt.timestep):  # still in free fall
      expected = com0 - 0.5 * 9.81 * 0.2**2
      assert abs(d.subtree_com[1][2] - expected) < 0.01
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    max_energy = max(max_energy, d.energy.sum())
    max_speed = max(max_speed, np.abs(d.qvel).max())
  assert max_energy < energy0 + 1.0, "energy grew: the simulation exploded"
  assert max_speed < 30.0
  # Ends collapsed on the floor, at rest, nothing below the floor.
  body_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
  lowest = min(d.geom_xpos[g][2] - m.geom_rbound[g] for g in body_geoms)
  assert lowest > -0.3  # rbound is a loose bound; real penetration is checked below
  for g in ("right_foot", "left_foot", "right_shin", "left_shin"):
    assert _capsule_min_z(m, d, g) > -0.02
  assert d.xpos[m.body("torso").id][2] < 0.6  # it fell down
  assert np.abs(d.qvel).max() < 0.5  # and came to rest


def test_standing_pose_rests_on_ground(model):
  m = model
  d = mujoco.MjData(m)
  mujoco.mj_resetDataKeyframe(m, d, m.keyframe("stand").id)
  mujoco.mj_forward(m, d)
  for foot in ("right_foot", "left_foot"):
    assert 0.0 <= _capsule_min_z(m, d, foot) < 0.002
    axis = d.geom_xmat[m.geom(foot).id].reshape(3, 3)[:, 2]
    assert abs(axis[2]) < 0.01  # sole is flat
  pelvis0 = d.xpos[m.body("torso").id][2]

  # Hold the pose with a PD controller for 1 s: feet stay planted, no sinking.
  # (Ankle stiffness must beat the gravity toppling stiffness m*g*h ~ 650 Nm/rad.)
  target = m.keyframe("stand").qpos[3:].copy()
  gear = m.actuator_gear[:, 0]
  floor = m.geom("floor").id
  feet = {m.geom("right_foot").id, m.geom("left_foot").id}
  for _ in range(int(1.0 / m.opt.timestep)):
    q, qd = d.qpos[3:], d.qvel[3:]
    d.ctrl[:] = np.clip((1000 * (target - q) - 30 * qd) / gear, -1, 1)
    mujoco.mj_step(m, d)
  touching = {c.geom1 if c.geom2 == floor else c.geom2 for c in d.contact[: d.ncon]}
  assert feet <= touching
  assert touching <= feet, "only the feet should touch the floor"
  for foot in ("right_foot", "left_foot"):
    assert _capsule_min_z(m, d, foot) > -0.005
  assert abs(d.xpos[m.body("torso").id][2] - pelvis0) < 0.03
  assert abs(d.qpos[2]) < 0.05  # torso still upright


@pytest.mark.parametrize("impl", ["warp", "jax"])
def test_random_actions_no_nan(model, impl):
  num_envs, num_steps, n_sub = 16, 1000, 10  # 1000 control steps = 25 s of sim
  mx = sprint.put_model(model, impl)
  key0 = model.keyframe("stand").qpos

  def init(key):
    qpos = jp.array(key0) + jax.random.uniform(key, (model.nq,), minval=-0.05, maxval=0.05)
    data = mjx.make_data(model, impl=impl, naconmax=num_envs * 16, njmax=80)
    return data.replace(qpos=qpos)

  @jax.jit
  def run(keys):
    data = jax.vmap(init)(keys)

    def control_step(carry, k):
      data = carry
      ctrl = jax.random.uniform(k, (num_envs, model.nu), minval=-1, maxval=1)

      def sub(data, _):
        return jax.vmap(lambda d, u: mjx.step(mx, d.replace(ctrl=u)))(data, ctrl), None

      data, _ = jax.lax.scan(sub, data, None, length=n_sub)
      bad = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
      return data, (bad, jp.abs(data.qvel).max())

    data, (bad, vmax) = jax.lax.scan(control_step, data, jax.random.split(keys[0], num_steps))
    return data, bad, vmax

  data, bad, vmax = run(jax.random.split(jax.random.PRNGKey(0), num_envs))
  assert not bool(bad.any()), f"NaN under impl={impl}"
  assert np.isfinite(np.asarray(data.qpos)).all()
  assert float(vmax.max()) < 100.0, f"velocities blew up under impl={impl}"
