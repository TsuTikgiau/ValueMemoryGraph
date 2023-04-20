from dm_control.mujoco import engine
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightSliderV0, \
    KitchenMicrowaveKettleBottomBurnerLightV0, KitchenBase


class KitchenMicrowaveKettleLightSliderV0Render(KitchenMicrowaveKettleLightSliderV0):
    def render(self, mode='human'):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            return super(KitchenBase, self).render()


class KitchenMicrowaveKettleBottomBurnerLightV0Render(KitchenMicrowaveKettleBottomBurnerLightV0):
    def render(self, mode='human'):
        if mode =='rgb_array':
            camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            img = camera.render()
            return img
        else:
            return super(KitchenBase, self).render()