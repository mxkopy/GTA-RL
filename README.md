### GTA-RL

![Test](GTA-RL.gif)

This (in progress!) repo is an attempt at making the streets of Los Santos safer from automotive injury. It uses [RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train a model to avoid crashing a car using reinforcement learning. The model architecture, training/visualization scripts, etc. are all found in the `scripts` folder. The graphics pipeline & engine hacking is in the `gta` folder.  

# How to run
1. Install [ScripthookV](https://www.dev-c.com/gtav/scripthookv/), [ViGEmBus](https://github.com/nefarius/ViGEmBus/releases), [PyTorch](https://pytorch.org/), then run `pip install -r requirements.txt`.
2. Copy `GTA-RL.asi` into the same folder as GTAV.exe. (Build instructions TBA)
3. Run `python scripts/main.py train` to start the model training thread. Metrics are logged via tensorboard in `ray_results`; run `tensorboard --logdir ray_results` to view them.  
4. Launch GTA and load into a singleplayer free-roam save.
5. [Optional] Run `python scripts/depth_buffer_view.py` to view the depth buffer (CTRL + C to quit).

# Some notes
This might not work with anti-aliasing or other post-processing effects, and if you resize the window things might break. You might have to experiment to find the right settings for your system. 

This has been done [a](https://arxiv.org/pdf/1712.01397) [bunch](https://arxiv.org/abs/1608.02192) [of times before](https://github.com/umautobots/GTAVisionExport/issues/13); this is largely an effort of modernizing some of these attempts :)

# Current bugs 
ViGEmBus will sometimes fail to launch at first within the model training process. This can be fixed by CTRL-Cing and restarting it until it works (though the goal is to replace it soon).

Sometimes things get initialized a little weirdly or out of order; one symptom of this is an error about a 'Payload' class or similar. This too can be fixed by CTRL-Cing and restarting. 