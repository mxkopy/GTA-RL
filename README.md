### GTA-RL

![Test](GTA-RL.gif)

This (in progress!) repo is an attempt at making the streets of Los Santos safer from automotive injury. It uses [RLlib](https://docs.ray.io/en/latest/rllib/index.html) to train a model to avoid crashing a car using reinforcement learning. 

# How to run
1. Install ScripthookV, ScripthookVDotNet, and ViGEmBus.
2. Install pytorch, and run `pip install -r requirements.txt`. 
3. Copy everything in the `build` folder into the game directory (i.e. where the .exe is located). 
4. Run `python main.py train` to start the model training thread. 
5. Launch GTA and load into a singleplayer free-roam save (make sure MSAA is disabled).
6. Press backspace.
7. [Optional] Install python-opencv & run `python depth_buffer_view.py` to view the depth buffer (CTRL + C to quit).

# Some notes
This might not work with anti-aliasing or other post-processing effects, and if you resize the window things might break. You might have to experiment to find the right settings for your system. 

This has been done before [a](https://arxiv.org/pdf/1712.01397) [bunch](https://arxiv.org/abs/1608.02192) of times, so I'm trying to do something a little new here. More specifically, I'm more focused on building out a framework for realtime RL in general than the performance of the model. This means the backend for passing messages between GTA V and the training program is pretty sophisticated (`ipc.py`, `ipc2.py`, `dxinterop/dllmain.cpp`, `dxinterop/ipc.h`) but the RL part is not as much, currently. 


# Python requirements 
torch, torchvision, cupy, numpy, protobuf, vgamepad

# Current bugs 
Protobuffers don't play well with GTA V, so I'm in the process of rewriting the IPC backend to use flatbuffers instead - the project won't work until then. 