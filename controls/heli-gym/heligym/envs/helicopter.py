"""
Helicopter control.
"""
import yaml
import sys, math
import numpy as np
import os
import copy

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding, EzPickle

from .dynamics import HelicopterDynamics
from .dynamics import WindDynamics
from .renderer.api import Renderer

FPS         = 50.0
DT          = 1/FPS
FTS2KNOT    = 0.5924838 # ft/s to knots conversion
EPS         = 1e-10 # small value for divison by zero
R2D         = 180/math.pi # Rad to deg
D2R         = 1/R2D
FT2MTR      = 0.3048 # ft to meter
TAU         = 2*math.pi
FLOAT_TYPE  = np.float32

class Heli(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }
    # Default maximum time for an episode
    default_max_time = 40.0
    # Default trim condition, ground trim.
    default_trim_cond = {
        "yaw": 0.0,
        "yaw_rate": 0.0,
        "ned_vel": [0.0, 0.0, 0.0],
        "gr_alt": 100.0,
        "xy": [0.0, 0.0],
        "psi_mr": 0.0,
        "psi_tr": 0.0
    }
    # default task target
    default_task_target = {}
    def __init__(self, heli_name:str = "aw109", render_enabled: bool = True):
        EzPickle.__init__(self)
        self.render_enabled = render_enabled
        yaml_path = os.path.join(os.path.dirname(__file__), "helis", heli_name + ".yaml")
        with open(yaml_path) as foo:
            params = yaml.safe_load(foo)

        self.heli_dyn = HelicopterDynamics(params, DT)
        self.wind_dyn = WindDynamics(params['ENV'], DT)
        self.heli_dyn.set_wind(self.wind_dyn.wind_mean_ned) # set mean wind as wind so that helicopter trims accordingly
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.heli_dyn.n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (self.heli_dyn.n_act,), dtype=np.float32)
        self.last_action = np.zeros(self.heli_dyn.n_act, dtype=np.float32)
        self.successed_time = 0 # time counter for successing task through time.
        self.set_max_time()
        self.set_target()
        self.set_trim_cond()
        self.set_reward_weights()
        self.normalizers = {
                                "t": np.sqrt(2*self.heli_dyn.MR["R"]/self.heli_dyn.ENV["GRAV"]),
                                "x": 2*self.heli_dyn.MR["R"],
                                "v": np.sqrt(2*self.heli_dyn.MR["R"]*self.heli_dyn.ENV["GRAV"]),
                                "a": self.heli_dyn.ENV["GRAV"]
                            }
        
        self._bGuiText = False
        if render_enabled:
            self.renderer = Renderer(w=1024, h=768, title='heligym')
            self.renderer.set_fps(FPS)

            self.heli_obj = self.renderer.create_model('/resources/models/'+heli_name+'/'+heli_name+'.obj',
                                                              '/resources/shaders/'+heli_name+'_vertex.vs',
                                                              '/resources/shaders/'+heli_name+'_frag.fs')
            self.renderer.add_permanent_object_to_window(self.heli_obj)

            self.terrain = self.renderer.create_model('/resources/models/terrain/terrain.obj',
                                                     '/resources/shaders/terrain_vertex.vs',
                                                     '/resources/shaders/terrain_frag.fs')
            self.renderer.add_permanent_object_to_window(self.terrain)

            self.sky = self.renderer.create_model('/resources/models/sky/sky.obj')
            self.renderer.add_permanent_object_to_window(self.sky)

            # Unit arrow along +X (NED North); length scaled by wind speed each frame.
            self.wind_arrow = self.renderer.create_model('/resources/models/arrow/arrow.obj')
            self.renderer.add_permanent_object_to_window(self.wind_arrow)

            # Cyan waypoint at position target (npos, epos, alt).
            self.target_marker = self.renderer.create_model('/resources/models/target/target.obj')
            self.renderer.add_permanent_object_to_window(self.target_marker)
        else:
            self.renderer = None
            self.heli_obj = None
            self.terrain = None
            self.sky = None
            self.wind_arrow = None
            self.target_marker = None

        # Hover setpoint in observation frame: npos/epos [ft], alt = -xyz[2] [ft].
        self.target_npos = float(self.default_trim_cond["xy"][0])
        self.target_epos = float(self.default_trim_cond["xy"][1])
        self.target_alt = float(self.default_trim_cond["gr_alt"])
        self._position_target_alt = self.target_alt
        self._gui_target_vals = np.zeros(3, dtype=np.float32)
    # Setter functions for RL tasks
    def set_max_time(self, max_time=None):
        self.max_time = self.default_max_time if max_time is None else max_time  # [sec] Given time for episode
        self.success_duration = self.max_time/4 # [sec] Required successfull time for maneuver
        self.task_duration = self.max_time/4 # [sec] Allowed time for successing task

    def set_target(self, target={}):
        self.task_target = self.default_task_target
        self.task_target.update(target)

    def get_target(self):
        return copy.deepcopy(self.task_target)

    def set_trim_cond(self, trim_cond={}):
        self.trim_cond = self.default_trim_cond
        self.trim_cond.update(trim_cond)
        xy = self.trim_cond.get("xy", [0.0, 0.0])
        default_alt = float(self.default_trim_cond["gr_alt"])
        self.set_position_target(
            npos=float(xy[0]),
            epos=float(xy[1]),
            alt=float(self.trim_cond.get("gr_alt", default_alt)),
        )

    def set_position_target(
        self,
        npos: float | None = None,
        epos: float | None = None,
        alt: float | None = None,
    ):
        """Set hover target in observation frame (npos/epos/alt, ft)."""
        if npos is not None:
            self.target_npos = float(npos)
        if epos is not None:
            self.target_epos = float(epos)
        if alt is not None:
            self.target_alt = float(alt)
            self._position_target_alt = self.target_alt

    def get_trim_cond(self):
        return copy.deepcopy(self.trim_cond)

    def set_reward_weights(self, base_reward_weight=None, terminal_reward_weight=None):
        zero_ = np.zeros((self.heli_dyn.n_obs,self.heli_dyn.n_obs))
        self.base_reward_weight = zero_ if base_reward_weight is None else base_reward_weight
        self.terminal_reward_weight = zero_ if terminal_reward_weight is None else terminal_reward_weight

    def __create_guiINFO_text(self):
        self.guiINFO_ID = self.renderer.create_guiText("Observations", 30.0, 30.0, 250.0, 0.0)
        self.guiINFO_text = []
        self.guiINFO_text.append("FPS        : %3.0f ")
        self.guiINFO_text.append("POWER      : %5.2f hp" )
        self.guiINFO_text.append("LON_VEL    : %5.2f ft/s")
        self.guiINFO_text.append("LAT_VEL    : %5.2f ft/s")
        self.guiINFO_text.append("DWN_VEL    : %5.2f ft/s")
        self.guiINFO_text.append("N_VEL      : %5.2f ft/s")
        self.guiINFO_text.append("E_VEL      : %5.2f ft/s")
        self.guiINFO_text.append("DES_RATE   : %5.2f ft/s")
        self.guiINFO_text.append("ROLL       : %5.2f rad")
        self.guiINFO_text.append("PITCH      : %5.2f rad")
        self.guiINFO_text.append("YAW        : %5.2f rad")
        self.guiINFO_text.append("ROLL_RATE  : %5.2f rad/sec")
        self.guiINFO_text.append("PITCH_RATE : %5.2f rad/sec")
        self.guiINFO_text.append("YAW_RATE   : %5.2f rad/sec")
        self.guiINFO_text.append("N_POS      : %5.2f ft")
        self.guiINFO_text.append("E_POS      : %5.2f ft")
        self.guiINFO_text.append("ALT        : %5.2f ft")
        self.guiINFO_text.append("GR_ALT     : %5.2f ft")

        self.guiACTION_ID = self.renderer.create_guiText("Agent Inputs", 320.0, 30.0, 250.0, 0.0)
        self.guiACTION_text = [
            "COLLECTIVE : %5.2f",
            "LON_CYCLIC : %5.2f",
            "LAT_CYCLIC : %5.2f",
            "PEDAL      : %5.2f",
        ]

        self.guiALT_ID = self.renderer.create_guiText("Altitude", 320.0, 180.0, 250.0, 0.0)
        self.guiALT_text = [
            "ALT (meas)   : %6.1f ft",
            "TARGET ALT   : %6.1f ft",
            "ALT ERROR    : %6.1f ft",
        ]
        self._gui_alt_vals = np.zeros(3, dtype=np.float32)

        self.guiTARGET_ID = self.renderer.create_guiText("Target", 320.0, 280.0, 250.0, 0.0)
        self.guiTARGET_text = [
            "N_POS        : %6.1f ft",
            "E_POS        : %6.1f ft",
            "ALT          : %6.1f ft",
        ]
        self._gui_target_vals = np.zeros(3, dtype=np.float32)

    def __add_to_guiText(self):
        info_val = np.array(self.renderer.get_fps())
        info_val = np.append(info_val, self.heli_dyn.observation)
        self.renderer.add_guiText(self.guiINFO_ID, self.guiINFO_text, info_val)
        self.renderer.add_guiText(self.guiACTION_ID, self.guiACTION_text, self.last_action)
        self.renderer.add_guiText(self.guiALT_ID, self.guiALT_text, self._gui_alt_vals)
        self._gui_target_vals[:] = [self.target_npos, self.target_epos, self.target_alt]
        self.renderer.add_guiText(self.guiTARGET_ID, self.guiTARGET_text, self._gui_target_vals)

    def _render_wind_arrow(self):
        """Place amber wind arrow above heli: length ∝ |W|, yaw toward W_NED."""
        if self.wind_arrow is None:
            return
        # Visual scale: meters of shaft per (ft/s) of wind; base sits above heli.
        arrow_up_ft = 25.0
        vis_scale = 0.15
        min_len_m = 0.5
        thick = 1.5

        wind = np.asarray(self.heli_dyn.WIND_NED, dtype=np.float64).reshape(3)
        spd = float(np.linalg.norm(wind))
        yaw = math.atan2(float(wind[1]), float(wind[0])) if spd > 1e-6 else 0.0
        length = max(min_len_m, spd * vis_scale)

        xyz = self.heli_dyn.state["xyz"]
        self.renderer.translate_model(
            self.wind_arrow,
            float(xyz[0]) * FT2MTR,
            float(xyz[1]) * FT2MTR,
            (float(xyz[2]) - arrow_up_ft) * FT2MTR,
        )
        # Earth-frame wind: yaw only (mesh +X = NED North).
        self.renderer.rotate_model(self.wind_arrow, 0.0, 0.0, yaw)
        # NED scale maps (x,y,z)->GL(x,-z,y); use z=-thick so GL Y stays positive
        # and face winding is not flipped under GL_CULL_FACE.
        self.renderer.scale_model(self.wind_arrow, length, thick, -thick)

    def _render_target_marker(self):
        """Cyan marker at hover target (npos, epos, alt) in NED/observation ft."""
        if self.target_marker is None:
            return
        # Observation: npos=xyz[0], epos=xyz[1], alt=-xyz[2].
        self.renderer.translate_model(
            self.target_marker,
            self.target_npos * FT2MTR,
            self.target_epos * FT2MTR,
            -self.target_alt * FT2MTR,
        )
        scale = 2.5
        self.renderer.scale_model(self.target_marker, scale, scale, -scale)

    def render(self):
        if self.renderer is None:
            return
        info_val = np.array(self.renderer.get_fps())
        info_val = np.append(info_val, self.heli_dyn.observation)
        
        self.renderer.set_guiText(self.guiINFO_ID, self.guiINFO_text, info_val)
        self.renderer.set_guiText(self.guiACTION_ID, self.guiACTION_text, self.last_action)

        alt = float(self.heli_dyn.observation[15])
        target_alt = float(self.target_alt)
        self._gui_alt_vals[:] = [alt, target_alt, alt - target_alt]
        self.renderer.set_guiText(self.guiALT_ID, self.guiALT_text, self._gui_alt_vals)

        self._gui_target_vals[:] = [self.target_npos, self.target_epos, self.target_alt]
        self.renderer.set_guiText(self.guiTARGET_ID, self.guiTARGET_text, self._gui_target_vals)

        self.renderer.rotate_MR(self.heli_obj, 
                                self.heli_dyn.state['betas'][1],
                                self.heli_dyn.state['betas'][0],
                                self.heli_dyn.state['psi_mr'][0])

        self.renderer.rotate_TR(self.heli_obj, 
                                0,
                                self.heli_dyn.state['psi_tr'][0],
                                0)


        self.renderer.translate_model(self.heli_obj, 
                                    self.heli_dyn.state['xyz'][0] * FT2MTR,
                                    self.heli_dyn.state['xyz'][1] * FT2MTR,
                                    self.heli_dyn.state['xyz'][2] * FT2MTR
                                    )

        self.renderer.rotate_model(self.heli_obj, 
                                    self.heli_dyn.state['euler'][0],
                                    self.heli_dyn.state['euler'][1],
                                    self.heli_dyn.state['euler'][2]
                                    )

        self.renderer.translate_model(self.sky, 
                                    self.heli_dyn.state['xyz'][0] * FT2MTR,
                                    self.heli_dyn.state['xyz'][1] * FT2MTR,
                                    self.heli_dyn.state['xyz'][2] * FT2MTR + 500
                                    )

        self._render_wind_arrow()
        self._render_target_marker()

        self.renderer.set_camera_pos(self.heli_dyn.state['xyz'][0] * FT2MTR,
                                     self.heli_dyn.state['xyz'][1] * FT2MTR + 30,
                                     self.heli_dyn.state['xyz'][2] * FT2MTR )


        if not self.renderer.is_visible():
            self.renderer.show_window()
            
        self.renderer.render()
        
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer.terminate()
    
    def exit(self):
        sys.exit()

    def step(self, actions):
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        self.last_action = np.asarray(actions, dtype=np.float32).reshape(-1)[: self.heli_dyn.n_act]
        self.time_counter += DT
        # Turbulence calculations
        pre_observations = self.heli_dyn.observation
        wind_action = np.concatenate([pre_observations[4:7], pre_observations[16:]])
        wind_turb_vel = self.wind_dyn.step(wind_action)
        # Helicopter dynamics calculations
        self.heli_dyn.set_wind(wind_turb_vel)  
        observation = self.heli_dyn.step(actions)
        reward, successed_step = self._calculate_reward()
        info = self._get_info()
        done = info['failed'] or info['successed'] or reward == np.nan
        truncated = info['time_up']
        self.successed_time += DT if successed_step else 0
        return np.copy(observation), reward, done, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            pass
        self.time_counter = 0
        self.successed_time = 0
        self.last_action.fill(0.0)
        self.wind_dyn.reset()
        self.heli_dyn.reset(self.trim_cond)
        self.set_position_target(
            npos=float(self.trim_cond.get("xy", [0.0, 0.0])[0]),
            epos=float(self.trim_cond.get("xy", [0.0, 0.0])[1]),
            alt=float(self.trim_cond["gr_alt"]),
        )
        if self.renderer is not None and not self._bGuiText:
            self.__create_guiINFO_text()
            self.__add_to_guiText()
            self._bGuiText = True
        return np.copy(self.heli_dyn.observation), self._get_info()

    def _get_info(self):
        return {
            'failed': self._is_failed(), 
            'successed': self._is_successed(), 
            'time_up': self._is_time_up()
            }

    def _is_failed(self):
        cond1 = self.heli_dyn._does_hit_ground(-self.heli_dyn.state['xyz'][2])
        cond2 = self.heli_dyn.state_dots['xyz'][2] > self.heli_dyn.MR['V_TIP']*0.05
        cond3 = self.heli_dyn.state['euler'][0] > 60*D2R
        cond4 = self.heli_dyn.state['euler'][1] > 60*D2R
        cond5 = np.abs(self.heli_dyn.state['xyz'][0]) > self.heli_dyn.ENV["NS_MAX"] / 2 or np.abs(self.heli_dyn.state['xyz'][1]) > self.heli_dyn.ENV["EW_MAX"] / 2 \
             or -self.heli_dyn.state['xyz'][2] > self.heli_dyn.ground_touching_altitude() + 10000
        cond = (cond1 and (cond2 or cond3 or cond4)) or cond5
        return cond

    def _is_successed(self):
        return self.successed_time >= self.success_duration  

    def _is_time_up(self):
        return self.time_counter > self.max_time

    def _calculate_reward(self):
        return 0.0, False

if __name__=='__main__':
    env = Heli()
    #print(env)
    #obs = env.reset()
    #action = np.array([0.7007, 0.5391, 0.5351, 0.6429])
    #obs, reward, done, info = env.step(action)
