from ipc import Channel
from struct import unpack, pack
import curses
import re
import numpy as np
import time
import math
from environment import VideoState

N = 3

VSBLengths = [Channel(8, f"VSB{i}Length") for i in range(N)]
VSBLen = lambda i: unpack("@Q", VSBLengths[i].pop_nbl())[0]
VSBFiles = [Channel(VSBLen(i), f"VSB{i}") for i in range(N)]
VSBView = lambda i, length=None, offset=0: VSBFiles[i].pop_nbl()[offset:(None if length is None else offset+length)]

def VSBViewType(i, n=None, offset=0, T='f'):
    if n is None:
        n = (VSBLen(i) - offset) // 4
    return unpack(f"@{n}{T}", VSBView(i, length=n*4, offset=offset))

def try_inverse():

    D = VideoState.pop_depth()[1, ...].squeeze()

    VBS = [np.array(VSBViewType(i)) for i in range(3)]
    MVP = np.asmatrix(VBS[2][28:28+16].reshape(4, 4))

    ROT = MVP
    ROT[:, 3] = 0

    SCREEN_X = VBS[2][4*15]
    SCREEN_Y = VBS[2][(4*15)+1]

# line 15: window size
def main(stdscr):

    stdscr.nodelay(True)

    R, G, B = Channel(12, "RVector"), Channel(12, "GVector"), Channel(12, "BVector")

    def writevec(vec: Channel, arr):
        vec.push_nbl(pack('@3f', *np.ravel(arr)))

    VAL_LEN = 10
    COL_LEN = 14

    n = 0

    while True:
        VBS = [np.array(VSBViewType(i)) for i in range(3)]
        MVP = np.asmatrix(VBS[2][28:28+16].reshape(4, 4))
    
        M1 = np.asmatrix(VBS[1][0:16].reshape(4, 4))
        M2 = np.asmatrix(VBS[1][16:32].reshape(4, 4))
        M3 = np.asmatrix(VBS[1][32:48].reshape(4, 4))

        ROT = MVP
        MVP[:, 3] = 0
        CAM = VBS[2][12:14]
        POS = VBS[2][16:18]
        SCREEN_X = VBS[2][4*15]
        SCREEN_Y = VBS[2][(4*15)+1]

        x, y = math.sin(n / 1000), math.cos(n / 1000)
        rv = np.ravel(np.array((x, 0, 0, y)) @ ROT) #* np.array(( 1914/1046, 1, 1, 1))
        gv = np.ravel(np.array((0, x, 0, y)) @ ROT) #* np.array(( 1914/1046, 1, 1, 1))
        bv = np.ravel(np.array((0, 0, x, y)) @ ROT) #* np.array(( 1914/1046, 1, 1, 1))

        writevec(R, rv[0:3])
        writevec(G, gv[0:3])
        writevec(B, bv[0:3])
        n = n + 1

        stdscr.clear()
        
        I = 2
        for r in range(len(VBS[I]) // 4):
            for c in range(4):
                stdscr.addstr(r, COL_LEN*c, str(VBS[I][4*r+c])[0:VAL_LEN])

        stdscr.addstr(r+1, 0, f'{SCREEN_X} {SCREEN_Y}')
        stdscr.refresh()

curses.wrapper(main)
