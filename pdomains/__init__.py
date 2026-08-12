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

# Same env with the spook alarm disabled, so the alarm's contribution to
# every gate can be measured separately.
register(
    id='pdomains-ant-tag-cdens-nospook-v0',
    entry_point='pdomains.ant_tag:CounterweightedDenAntTagEnv',
    max_episode_steps=300,
    kwargs={'spook': False},
)
