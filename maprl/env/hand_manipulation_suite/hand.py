import mujoco_py
from d4rl.hand_manipulation_suite.pen_v0 import PenEnvV0
from d4rl.hand_manipulation_suite.hammer_v0 import HammerEnvV0
from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from d4rl.hand_manipulation_suite.relocate_v0 import RelocateEnvV0


class PenEnvV0Render(PenEnvV0):
    def viewer_setup(self):
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.distance = 1.8031948963187563
        self.viewer.cam.lookat[0] = 0.00650679
        self.viewer.cam.lookat[1] = -0.17454417
        self.viewer.cam.lookat[2] = 0.2005649
        self.viewer.cam.azimuth = -45.0
        self.viewer.cam.elevation = -45.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            if not hasattr(self, 'viewer'):
                self.viewer_setup()
            self.viewer.render(2560, 1920, camera_id=0)
            img = self.viewer.read_pixels(2560, 1920, depth=False)
            return img[::-1]
        else:
            return super(PenEnvV0Render, self).render(mode)


class HammerEnvV0Render(HammerEnvV0):
    def viewer_setup(self):
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.distance = 1.8031948963187563
        self.viewer.cam.lookat[0] = 0.00650679
        self.viewer.cam.lookat[1] = -0.17454417
        self.viewer.cam.lookat[2] = 0.2005649
        self.viewer.cam.azimuth = -45.0
        self.viewer.cam.elevation = -45.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            if not hasattr(self, 'viewer'):
                self.viewer_setup()
            self.viewer.render(2560, 1920, camera_id=0)
            img = self.viewer.read_pixels(2560, 1920, depth=False)
            return img[::-1]
        else:
            return super(HammerEnvV0Render, self).render(mode)


class DoorEnvV0Render(DoorEnvV0):
    def viewer_setup(self):
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.distance = 1.8031948963187563
        self.viewer.cam.lookat[0] = 0.00650679
        self.viewer.cam.lookat[1] = -0.17454417
        self.viewer.cam.lookat[2] = 0.2005649
        self.viewer.cam.azimuth = -45.0
        self.viewer.cam.elevation = -45.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            if not hasattr(self, 'viewer'):
                self.viewer_setup()
            self.viewer.render(2560, 1920, camera_id=0)
            img = self.viewer.read_pixels(2560, 1920, depth=False)
            return img[::-1]
        else:
            return super(DoorEnvV0Render, self).render(mode)


class RelocateEnvV0Render(RelocateEnvV0):
    def viewer_setup(self):
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)
        self.viewer.cam.distance = 1.8031948963187563
        self.viewer.cam.lookat[0] = 0.00650679
        self.viewer.cam.lookat[1] = -0.17454417
        self.viewer.cam.lookat[2] = 0.2005649
        self.viewer.cam.azimuth = -45.0
        self.viewer.cam.elevation = -45.0

    def render(self, mode='human'):
        if mode == 'rgb_array':
            if not hasattr(self, 'viewer'):
                self.viewer_setup()
            self.viewer.render(2560, 1920, camera_id=0)
            img = self.viewer.read_pixels(2560, 1920, depth=False)
            return img[::-1]
        else:
            return super(RelocateEnvV0Render, self).render(mode)
