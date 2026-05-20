from changwon_env import ChangwonRoutingEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import time, multiprocessing

NUM_ENVS = 16
TOTAL_TIMESTEPS = 2_000_000

class ProgressLogger(BaseCallback):
    """Logs progress to file every 20k steps."""
    def __init__(self, log_path="training.log", print_every=20000):
        super().__init__()
        self.log_path = log_path
        self.print_every = print_every
        self.start_time = None
        self.last_print = 0
        self.logfile = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.logfile = open(self.log_path, "w", buffering=1)
        msg = f"[START] Training begun at {time.strftime('%H:%M:%S')} | {NUM_ENVS} parallel envs | MaskablePPO + GPU\n"
        self.logfile.write(msg)
        print(msg, end="", flush=True)

    def _on_step(self):
        steps = self.num_timesteps
        if steps - self.last_print >= self.print_every:
            elapsed = time.time() - self.start_time
            fps = steps / elapsed if elapsed > 0 else 0
            pct = steps / TOTAL_TIMESTEPS * 100
            eta = (TOTAL_TIMESTEPS - steps) / fps if fps > 0 else 0
            msg = f"[{pct:5.1f}%] {steps:>10,}/{TOTAL_TIMESTEPS:,} | {elapsed:6.0f}s | {fps:,.0f} fps | ETA {eta:.0f}s\n"
            self.logfile.write(msg)
            print(msg, end="", flush=True)
            self.last_print = steps
        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time
        msg = f"[DONE] Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)\n"
        self.logfile.write(msg)
        print(msg, end="", flush=True)
        self.logfile.close()

def make_env():
    def _init():
        return ChangwonRoutingEnv(num_nodes=20)
    return _init

def train_ppo(total_timesteps=TOTAL_TIMESTEPS):
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    model = MaskablePPO(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=3e-4,
        batch_size=512,
        n_steps=2048,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # encourage exploration
        policy_kwargs=dict(net_arch=[256, 256]),  # larger network
        device="cuda",
    )
    print(f"Training MaskablePPO (2M steps) | {NUM_ENVS} envs | batch=512 | net=[256,256] | GPU")
    callback = ProgressLogger(log_path="training.log", print_every=20000)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("ppo_changwon_routing_deep")
    print("Model saved to ppo_changwon_routing_deep.zip")
    env.close()
    return model

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    train_ppo()
