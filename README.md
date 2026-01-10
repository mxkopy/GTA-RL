### GTA-RL

![Test](GTA-RL.gif)

This (in progress!) repo is an attempt at making the streets of Los Santos safer from automotive injury. It implements Deep Deterministic Policy Gradient a la [OpenAI](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) to train a model to avoid crashing a car. 

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

# Python requirements 
torch, torchvision, cupy, numpy, protobuf, vgamepad
