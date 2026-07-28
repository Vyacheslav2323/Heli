import numpy as np
import heligym
from pynput import keyboard

# Per-frame control rates (normalized action space)
COLLECTIVE_RATE = 0.02
PEDAL_RATE = 0.02
CYCLIC_RATE = 0.03

pressed_keys = set()
running = True
trim_action = None
action = None
gui_controls_id = None
gui_controls_text = [
    "COLLECTIVE : %6.3f",
    "LON_CYCLIC : %6.3f",
    "LAT_CYCLIC : %6.3f",
    "PEDAL      : %6.3f",
]


def _key_char(key, char):
    return getattr(key, "char", None) and key.char.lower() == char


def on_press(key):
    global running, action, trim_action
    if key == keyboard.Key.esc:
        running = False
        return False
    if _key_char(key, "r") and trim_action is not None:
        action = trim_action.copy()
    pressed_keys.add(key)


def on_release(key):
    pressed_keys.discard(key)


def update_action(current):
    """Apply held keys to the persistent control vector."""
    updated = current.copy()
    for key in pressed_keys:
        if _key_char(key, "w"):
            updated[0] += COLLECTIVE_RATE
        elif _key_char(key, "s"):
            updated[0] -= COLLECTIVE_RATE
        elif _key_char(key, "a"):
            updated[3] -= PEDAL_RATE
        elif _key_char(key, "d"):
            updated[3] += PEDAL_RATE
        elif key == keyboard.Key.up:
            updated[1] += CYCLIC_RATE
        elif key == keyboard.Key.down:
            updated[1] -= CYCLIC_RATE
        elif key == keyboard.Key.left:
            updated[2] -= CYCLIC_RATE
        elif key == keyboard.Key.right:
            updated[2] += CYCLIC_RATE
    return np.clip(updated, -1.0, 1.0).astype(np.float32)


def reset_episode(env):
    global trim_action, action
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result
    trim_action = env.hetrim_action = np.zeros(4, dtype=np.float32)
    action = trim_action.copy()
    return obs


def print_controls():
    print(
        "Controls: W/S collective | A/D pedal | arrows cyclic | R reset trim | Esc quit"
    )


def setup_controls_gui(env, initial_action):
    global gui_controls_id
    gui_controls_id = env.renderer.create_guiText("Controls", 300.0, 30.0, 280.0, 0.0)
    env.renderer.add_guiText(gui_controls_id, gui_controls_text, initial_action)


def update_controls_gui(env, current_action):
    env.renderer.set_guiText(gui_controls_id, gui_controls_text, current_action)


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

env = heligym.HeliHover()
obs = reset_episode(env)
setup_controls_gui(env, action)
print_controls()

try:
    while running and not env.renderer.is_close():
        # Four controls:
        # [collective, longitudinal cyclic, lateral cyclic, pedal]
        action = update_action(action)
        update_controls_gui(env, action)
        env.render()

        result = env.step(action)

        # Old Gym API returns: obs, reward, done, info
        # Newer wrappers may return 5 values
        if len(result) == 4:
            obs, reward, done, info = result
            if done:
                obs = reset_episode(env)
        else:
            obs, reward, terminated, truncated, info = result
            if terminated or truncated:
                obs = reset_episode(env)
finally:
    listener.stop()
    env.close()
