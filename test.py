import numpy as np
# import torch
# import torch.nn as nn
import cvxpy as cvx
import pickle
import time

from graphics_test import ShaderVars, try_raycast, save_raycast, load_raycast, DummyShader, quiver, show

def save_rays(n=50):
    def flatten_raycast(raycastinfo):
        rays = raycastinfo['Rays']
        del raycastinfo['Rays']
        for ray in rays:
            ray |= raycastinfo
        return rays
    with open('rays.pickle', 'wb') as file:
        rays = flatten_raycast(try_raycast())
        for _ in range(n):
            ray = try_raycast()
            while (rays[-1]['P'] == ray['P']).all():
                ray = try_raycast()            
            rays = rays + flatten_raycast(ray)
        pickle.dump(rays, file)

def load_rays():
    with open('rays.pickle', 'rb') as file:
        return pickle.load(file)

def get_oddities(ray):
    VSB1 = np.array(ray['VSB1']).reshape(-1, 4)
    VSB2 = np.array(ray['VSB2']).reshape(-1, 4)
    A, B, C = VSB1[4:8, :], VSB1[8:12, :], VSB1[12:, :]
    L, R = VSB2[12, -1], VSB2[13, -1]
    n, f = ray['NearClip'], ray['FarClip']
    # print(n/f,  -(n+f)/(f-n), (-2*f*n)/(f-n))
    # print(ray['NearClip'])
    # print(ray['FarClip'])
    # print(VSB2)
    # exit()
    X = (VSB2[7, :3] - VSB2[8, :3]) / np.linalg.norm(VSB2[8, :3] - VSB2[7, :3])
    Y = (VSB2[10, :3] - VSB2[9, :3]) / np.linalg.norm(VSB2[10, :3] - VSB2[9, :3])
    Z = (VSB2[7, :3] + VSB2[8, :3]) / np.linalg.norm(VSB2[7, :3] + VSB2[8, :3])
    ROT = np.stack([X, Y, Z], axis=0)
    return {
        'A': A,
        'B': B,
        'C': C,
        'L': L,
        'R': R,
        'ROT': ROT,
        'SCREEN_X': VSB2[15, 0],
        'SCREEN_Y': VSB2[15, 1]
    }


# save_rays()
rays = load_rays()

# VSB2 = np.array(rays[-1]['VSB2']).reshape(-1, 4)
# for i, row in enumerate(VSB2):
    # print(i, row)
# exit()

def to_view_ray(ray, oddity):
    x = ray['pixel'][0] / oddity['SCREEN_X']
    y = ray['pixel'][1] / oddity['SCREEN_Y']
    z = 1/ray['depth']
    w = 1.0
    return np.array((2.0*x-1.0, 2.0*y-1.0, z, w))

collisions = [np.array(ray['collision']) - np.array(ray['P']) for ray in rays]
true_depths = [np.linalg.norm(collision) for collision in collisions]
depths = [ray['depth'] for ray in rays]
oddities = [get_oddities(ray) for ray in rays]
view_rays = [to_view_ray(ray, oddity) for ray, oddity in zip(rays, oddities)]
perspectives = [oddity['C'] @ view_ray for oddity, view_ray in zip(oddities, view_rays)]
depth_sorted_idx = sorted(range(len(rays)), key=lambda i: true_depths[i])

# print(rays[depth_sorted_idx[0]]['pixel'])
# print(rays[depth_sorted_idx[0]]['depth'])

# exit()

# for t, d in zip(true_depths, depths):
#     z = 2*d-1

#     print(1/(t*z))

# exit()
nf = lambda ray: (ray['NearClip'] + ray['FarClip'])/(ray['FarClip'] - ray['NearClip'])
normalize = lambda A : (A - A.min()) / (A.max() - A.min())
def produce_arrays():
    X, Y, P, N, F= [], [], [], [], []
    for ray, t, d, p, oddity in zip(rays, true_depths, depths, perspectives, oddities):
        n, f, l, r = ray['NearClip'], ray['FarClip'], oddity['L'], oddity['R']
        # a, b = f/(f-n), (-f*n)/(f-n)
        x = t
        # y = 1/((d+1)/2)
        y = 1/pow(f, d)
        
        # 1/((f/(f+n))+d)
        # y = d
        p = p[-1]
        X.append(x)
        Y.append(y)
        P.append(p)
        N.append(n)
        F.append(f)
    X = np.array(X)
    Y = np.array(Y)
    P = np.array(P)
    N = np.array(N)
    F = np.array(F)
    return X[depth_sorted_idx], Y[depth_sorted_idx], P[depth_sorted_idx], N[depth_sorted_idx], F[depth_sorted_idx]

from matplotlib import pyplot as plt
x, y, p, n, f = produce_arrays()
x, y, f = normalize(x), normalize(y), normalize(f) 
plt.plot(x, color='blue')
plt.plot(y, color='red')
plt.plot(f, color='purple')
plt.plot()
plt.show()
exit()


# for i in range(0, 4):
#     A = oddities[i]['A']
#     B = oddities[i]['B']
#     C = oddities[i]['C']
#     ROT = oddities[i]['ROT']
#     collision = collisions[i]
#     # collision = collision / np.linalg.norm(collision)
#     collision = ROT @ collision
#     # direction = collision / np.linalg.norm(direction)
#     collision = np.append(collision, 1)
#     P = C @ collision
#     P /= P[-1]
#     eye = np.eye(3)
#     quiver(eye[0], color='black')
#     quiver(eye[1], color='black')
#     quiver(eye[2], color='black')
#     quiver(P[:3])
#     # quiver(A @ collision)
#     # quiver(ROT @ collision)
#     # quiver(ROT.T @ collision)
#     # quiver(ROT[0], color='black')
#     # quiver(ROT[1], color='black')
#     # quiver(ROT[2], color='black')
#     # quiver(collision / np.linalg.norm(collision))
# show(block=True)
# exit()

# VSB1 = np.array(raycastinfo['VSB1']).reshape(-1, 4)
# VSB2 = np.array(raycastinfo['VSB2']).reshape(-1, 4)
# A, B, C = VSB1[4:8, ...], VSB1[8:12, ...], VSB1[12:, ...],

# U = VSB2[12, -2]
# D = VSB2[13, -1]

slacks = [cvx.Variable() for _ in depths]

A = cvx.Variable()
B = cvx.Variable()

constraints = [(B-A*d) == (1/t)+s for (d, t, s) in zip(y, x, slacks)]

objective = cvx.sum([cvx.abs(slack) for slack in slacks])
problem = cvx.Problem(cvx.Minimize(objective), constraints=constraints)
problem.solve(max_iter=1000)
A, B = A.value, B.value
print(problem.value)
print(A, B)
for d, t in zip(y, x):
    print(1/(B - A*d), t)