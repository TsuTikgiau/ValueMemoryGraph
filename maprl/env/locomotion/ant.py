from d4rl.locomotion.ant import AntMazeEnv
from d4rl.locomotion import wrappers


class AntMazeEnvRender(AntMazeEnv):
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.2
        self.viewer.cam.lookat[0] += 5
        self.viewer.cam.lookat[1] += 5
        self.viewer.cam.elevation = -90


def make_ant_maze_env(**kwargs):
    env = AntMazeEnvRender(**kwargs)
    return wrappers.NormalizedBoxEnv(env)
