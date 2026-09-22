from gymnasium.envs.registration import register

register(
    id='pdomains-car-flag-v0',
    entry_point='pdomains.car_flag:CarEnv',
    max_episode_steps=160,
)

register(
    id='pdomains-two-boxes-v0',
    entry_point='pdomains.two_boxes:BoxEnv',
    max_episode_steps=100,
)

register(
    id='pdomains-ant-heaven-hell-v0',
    entry_point='pdomains.ant_heaven_hell:AntEnv',
    max_episode_steps=400,
)

register(
    id='pdomains-ant-tag-v0',
    entry_point='pdomains.ant_tag:AntTagEnv',
    max_episode_steps=400,
)

register(
    id='pdomains-ant-tag-smart-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
)

# SmartAntTag with the cdens-hard / cdens-terminal sensing and tagging
# geometry (tag 0.6, visible 1.0 instead of 1.5 / 3.0), and nothing else
# changed: same 9x9 cage, same evading target, same 400-step cap. Isolates
# "how much of the counterweighted-den difficulty is just the radii?" from
# the den structure, and stresses belief tracking (search / re-acquisition)
# rather than the terminal chase.
register(
    id='pdomains-ant-tag-smart-hard-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
    kwargs={
        'tag_radius': 0.6,
        'visible_radius': 1.0,
    },
)

# Geometry sweep around smart-hard (2026-09-05). The fully-observed PPO ceiling
# on smart-hard is only 15-42% at 3M steps: a 0.6 tag radius against a target
# stepping 0.5 is a hard chase even with perfect information. These three
# relax the chase along the two available axes while keeping the target
# partially observed (visible 2.0 on the 9x9 cage is still only ~16% of the
# arena). Same class, same cap; kwargs only.
register(  # bigger radii, same target speed
    id='pdomains-ant-tag-smart-mid-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
    kwargs={'tag_radius': 1.0, 'visible_radius': 2.0},
)
register(  # smart-hard radii, slower target
    id='pdomains-ant-tag-smart-hard-slow-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
    kwargs={'tag_radius': 0.6, 'visible_radius': 1.0, 'target_step': 0.3},
)
register(  # both
    id='pdomains-ant-tag-smart-mid-slow-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
    kwargs={'tag_radius': 1.0, 'visible_radius': 2.0, 'target_step': 0.3},
)
register(  # smart-mid-slow with a tighter visible radius (8.7% of the cage)
    id='pdomains-ant-tag-smart-mid-slow-v15-v0',
    entry_point='pdomains.ant_tag:SmartAntTagEnv',
    max_episode_steps=400,
    kwargs={'tag_radius': 1.0, 'visible_radius': 1.5, 'target_step': 0.3},
)

register(
    id='pdomains-ant-tag-ghost-v0',
    entry_point='pdomains.ant_tag:GhostAntTagEnv',
    max_episode_steps=400,
)

# 200, NOT 400: the tight episode cap is the mechanism that makes committing
# to the wrong den expensive (a wasted ~46-step den-to-den crossing plus a
# badly-ordered chase overruns it). See the twin-den plan section 1.2.
register(
    id='pdomains-ant-tag-dens-v0',
    entry_point='pdomains.ant_tag:TwinDenAntTagEnv',
    max_episode_steps=200,
)
# 300, NOT 200/400: crossing the h+f = 9.15-unit den separation takes ~55
# steps at the ant's ~0.165/step, so informed play fits in ~40-130 steps while
# a wasted crossing plus post-spook pursuit usually does not. See the
# counterweighted-den plan section 1.2.
register(
    id='pdomains-ant-tag-cdens-v0',
    entry_point='pdomains.ant_tag:CounterweightedDenAntTagEnv',
    max_episode_steps=300,
)

# Harder sensing/tagging geometry. Keep the original id unchanged so its
# completed runs remain exactly reproducible. The spook radius equals the
# first possible visual-overlap distance: visible_radius + cden_r = 1.4.
register(
    id='pdomains-ant-tag-cdens-hard-v0',
    entry_point='pdomains.ant_tag:CounterweightedDenAntTagEnv',
    max_episode_steps=300,
    kwargs={
        'tag_radius': 0.6,
        'visible_radius': 1.0,
        'spook_radius': 1.4,
    },
)

# Irreversible-commitment variant of the hard counterweighted-den task.
# The two candidate dens belonging to the inactive mirrored arrangement are
# terminal hazards.  This removes the Gaussian policy's ability to inspect
# both mirrored heavy candidates sequentially: crossing a phantom den's
# earliest-information boundary ends the episode with a penalty.  A centered
# spawn keeps a direct commitment to the active heavy den from accidentally
# crossing a phantom region merely because of the initial ant position.
register(
    id='pdomains-ant-tag-cdens-terminal-v0',
    entry_point='pdomains.ant_tag:TerminalPhantomCounterweightedDenAntTagEnv',
    max_episode_steps=300,
    kwargs={
        'tag_radius': 0.6,
        'visible_radius': 1.0,
        'spook_radius': 1.4,
        'phantom_terminal_radius': 1.4,
        # At least as costly as exhausting the 300-step horizon.  This keeps
        # fast deliberate failure from looking better than a slower tag to
        # both PPO and EvalCallback's return-based best-model selection.
        'phantom_penalty': -300.0,
        'central_spawn_radius': 0.5,
    },
)

# Same env with the spook alarm disabled, so the alarm's contribution to
# every gate can be measured separately.
register(
    id='pdomains-ant-tag-cdens-nospook-v0',
    entry_point='pdomains.ant_tag:CounterweightedDenAntTagEnv',
    max_episode_steps=300,
    kwargs={'spook': False},
)

# --- Odd-Even POMDP -------------------------------------------------------
# One hidden integer state, drawn uniformly from [1, n] at reset and static
# for the episode. Every step emits noisy same-parity observations, so the
# belief is a categorical posterior that only ever sharpens.
#
# obs_per_step MUST stay 1. The observations are i.i.d. given the state, so k
# per step shrink the standard error by sqrt(k): at n=50 the env's old default
# of 100 located the state to +/-0.28 against a grid of spacing 1, which
# solves the task at step 1 and makes every belief encoder score alike.
#
# The caps are the point of the three ids, so they are set here and nowhere
# else -- a script that defaults its own cap instead of reading
# max_episode_steps re-runs the Ant-Tag mistake where a 400-step eval default
# against a 200-step env counted every timeout as a success.

# 50: the exact posterior reaches max(belief) > 0.9 on the true state by about
# step 14 at n=10, so the transient is ~28% of the episode -- the cheap id.
register(
    id='pdomains-odd-even-10-v0',
    entry_point='pdomains.odd_even_pomdp:make_odd_even_pomdp',
    max_episode_steps=50,
    kwargs={
        'n_dist_size': 10,
        'obs_per_step': 1,
    },
)

# 50: the main case. std_dev = sqrt(50)/sqrt(10) + 1 = 3.236 spans about three
# same-parity neighbours on each side, so the belief locks on by about step 21
# and the transient is ~42% of the episode -- the widest window in which two
# encoders can still differ.
register(
    id='pdomains-odd-even-50-v0',
    entry_point='pdomains.odd_even_pomdp:make_odd_even_pomdp',
    max_episode_steps=50,
    kwargs={
        'n_dist_size': 50,
        'obs_per_step': 1,
    },
)

# 200, NOT 50: the steady-state stress id. About 90% of the episode sits in
# the near-one-hot regime where every encoder is equivalent, which is what
# makes it the control for pooled averaging -- the oracle's own reward per
# step moves from -1.098 to -0.582 between a 50- and a 100-step cap purely
# from transient dilution, so a single pooled mean is not comparable across
# caps.
register(
    id='pdomains-odd-even-50-long-v0',
    entry_point='pdomains.odd_even_pomdp:make_odd_even_pomdp',
    max_episode_steps=200,
    kwargs={
        'n_dist_size': 50,
        'obs_per_step': 1,
    },
)

# 30 steps, NOT 50: the encoder comparison lives in the TRANSIENT. A Gaussian
# (mean + variance) encoding of the belief is blind to the state's parity only
# while the belief is still WIDE -- measured, rounding the belief mean recovers
# parity at 0.357 after 2 observations, 0.512 after 8, 0.845 after 30. Once the
# posterior collapses onto one state its mean sits ON that state, so parity
# comes free and every encoding converges. A 50-step cap spends most of the
# episode in that collapsed regime and averages the effect away; 30 keeps the
# comparison inside the window where the encodings actually differ.
register(
    id='pdomains-odd-even-50-short-v0',
    entry_point='pdomains.odd_even_pomdp:make_odd_even_pomdp',
    max_episode_steps=30,
    kwargs={'n_dist_size': 50, 'obs_per_step': 1},
)

# ---------------------------------------------------------------------------------------------
# Hunt tasks (2026-09-13): Cluster-Hunt, least-mass and most-var, moved from the parent repo's
# src/hunt_tasks/env/ (see pdomains/hunt.py). Each env truncates itself at max_steps = 60, and
# the registration cap says the same so the harness reads it off gym.spec().
# ---------------------------------------------------------------------------------------------
register(
    id='pdomains-cluster-hunt-v0',
    entry_point='pdomains.hunt:ClusterHuntEnv',
    max_episode_steps=60,
    # The recorded RL configuration (src/hunt_tasks/train.py defaults; cluster_hunt.md sec. 7).
    kwargs=dict(hit_radius=0.6, min_sep=2.5, max_steps=60),
)

register(
    id='pdomains-least-mass-v0',
    entry_point='pdomains.hunt:MinMassHuntEnv',
    max_episode_steps=60,
)

register(  # the widest cluster is the target; widths in [0.30, 0.90], unique by a 0.2 margin
    id='pdomains-most-var-v0',
    entry_point='pdomains.hunt:MinMassHuntEnv',
    max_episode_steps=60,
    kwargs=dict(target_rule="max_var", sigma_hi=0.9, sigma_margin=0.2),
)

# Multimodal Search (2026-09-13): moved from set_transformer/rl/envs/multimodal_search.py (see
# pdomains/multimodal_search.py). Default configuration; the env truncates itself at max_steps = 42.
register(
    id='pdomains-multimodal-search-v0',
    entry_point='pdomains.multimodal_search:make_env',
    max_episode_steps=42,
)

# Light-Dark 1D (2026-09-19): Python port of LightDark1D from POMDPs.jl (see
# pdomains/light_dark.py). Defaults are the Julia LightDark1D() constructor. The episode
# cap lives here: the env itself never truncates, matching the unbounded Julia problem.
# 50 steps is comfortably above the ~15 an optimal detour (localise at the light at y=5,
# return to the goal at y=0, declare) needs from y0 ~ N(2, 3).
register(
    id='pdomains-light-dark-1d-v0',
    entry_point='pdomains.light_dark:make_env',
    max_episode_steps=50,
)
