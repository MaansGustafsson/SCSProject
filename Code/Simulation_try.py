import math
import numpy as np 
import matplotlib
from matplotlib.animation import FuncAnimation, FFMpegWriter
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

#How many points in each area
N = 100
#Total points (has to be 3xN)
N2 = 300

#Arena proportions
Lx = 140
Ly = 40

#Velocity and size of time step
v0 = 4
dt = 0.01
exit_cooldown = int(0.5/dt) # Two exits every seconds

#Max run steps
t_tot = 50000

#The maximum amount of random rotation
rotation_range = np.pi / 10

#Initialize random positions in 3 different areas of the arena
x = np.zeros(N2)
y = np.zeros(N2)
x[:N] = np.random.randint(low = 31, high = 104, size=[N])
y[:N] = np.random.randint(low = 29, high = 39, size=[N])
x[N:2*N] = np.random.randint(low = 105, high = 139, size=[N])
y[N:2*N] = np.random.randint(low = 1, high = 39, size=[N])
x[2*N:] = np.random.randint(low = 1, high = 43, size=[N])
y[2*N:] = np.random.randint(low = 10, high = 27, size=[N])

#----------- CHOSE SIMULATION TYPE --------#
sim = 1 # sim = 0 is closest door, sim = 1 is randomly assigned door, sim = 2 is with some 'drunk' particles

# Center points for doors
door_1 = [14.5, 28]  #(nx>12) & (nx<17) & (ny>=28)
door_2 = [106.5, 41] #(nx>104) & (nx<109) & (ny>=40)
door_3 = [141, 7.5]  #(ny>5) & (ny<10) & (nx>=140)
doors = [door_1,door_2, door_3]

rand_door = np.random.randint(low=0,high=3,size=N2)

#Randomizes angle of velocity
phi = np.random.rand(N2)*2*np.pi
vx = v0*np.cos(phi)
vy = v0*np.sin(phi)

x_path = np.zeros([t_tot,N2])
y_path = np.zeros([t_tot,N2])

exit1_time = 0
exit2_time = 0
exit3_time = 0

def find_closest_door(x, y, doors):
    doors_x = np.array([door[0] for door in doors])
    doors_y = np.array([door[1] for door in doors])
    closest_door = []

    for p in range(len(x)):
        dx = doors_x - x[p]
        dy = doors_y - y[p]
        dist = np.sqrt(dx**2 + dy**2)
        closest = np.argmin(dist)  # Find index of closest door
        closest_door.append(closest)
    return(closest_door)            

def point_to_door(x, y, chosen_door, doors):
    doors_x = np.array([door[0] for door in doors])
    doors_y = np.array([door[1] for door in doors])
    vx = []
    vy = []

    for i in range(len(x)):
        door = chosen_door[i]
        dx = doors_x[door] - x[i]
        dy = doors_y[door] - y[i]
        dist = np.sqrt(dx**2 + dy**2) + 1e-16  # Avoid division by zero
        
        # Normalize velocity and scale by v0
        vx.append(v0 * (dx / dist))
        vy.append(v0 * (dy / dist))

    return np.array(vx), np.array(vy)


for t in range(t_tot):

    #Saves the path of the points
    x_path[t,:] = x
    y_path[t,:] = y
    
    # Find the closest door for each particle
    closest_door = find_closest_door(x, y, doors)

    #Different velocities depending on which simulation is ran
    if sim == 0:
        vx, vy = point_to_door(x, y, closest_door, doors)
        #Updates the velocities with random angle
        randangle = (np.random.rand(N2) - 0.5) * 2 * rotation_range
        nvx = np.cos(randangle) * vx - np.sin(randangle) * vy
        nvy = np.sin(randangle) * vx + np.cos(randangle) * vy
    elif sim == 1:
        vx, vy = point_to_door(x, y, rand_door, doors)
        #Updates the velocities with random angle
        randangle = (np.random.rand(N2) - 0.5) * 2 * rotation_range
        nvx = np.cos(randangle) * vx - np.sin(randangle) * vy
        nvy = np.sin(randangle) * vx + np.cos(randangle) * vy
    else:
        drunk_range = rotation_range*6
        vx, vy = point_to_door(x, y, closest_door, doors)
        #Updates the velocities with increased random angle
        randangle_drunk = (np.random.rand(N) - 0.5) * 2 * drunk_range
        randangle = (np.random.rand(N2) - 0.5) * 2 * rotation_range
        nvx = np.cos(randangle) * vx - np.sin(randangle) * vy
        nvy = np.sin(randangle) * vx + np.cos(randangle) * vy
        nvx[int(N/2):int(3*N/2)] = np.cos(randangle_drunk) * vx[int(N/2):int(3*N/2)] - np.sin(randangle_drunk) * vy[int(N/2):int(3*N/2)]
        nvy[int(N/2):int(3*N/2)] = np.sin(randangle_drunk) * vx[int(N/2):int(3*N/2)] + np.cos(randangle_drunk) * vy[int(N/2):int(3*N/2)]       

    #Studsande partiklar




    #Redirect around walls
    ind_wall1 = np.where((x<31) & (y>=28))
    if np.size(ind_wall1) > 0:
        nvy[ind_wall1] = nvx[ind_wall1]
        #nvx[ind_wall1] = abs(nvx[ind_wall1])

    ind_wall2 = np.where((x<105) & (x>103) & (y<=30))
    if np.size(ind_wall2) > 0:
        nvy[ind_wall2] = -nvx[ind_wall2]
        #nvx[ind_wall2] = abs(nvx[ind_wall2])

    ind_wall3 = np.where((x<69) & (x>66) & (y<=30))
    if np.size(ind_wall3) > 0:
        nvy[ind_wall3] = nvx[ind_wall3]
        #nvx[ind_wall3] = -(nvx[ind_wall3])

    ind_wall4 = np.where((x<45) & (x>43) & (y<=21))
    if np.size(ind_wall4) > 0:
        nvy[ind_wall4] = nvx[ind_wall4]
        #nvx[ind_wall4] = -(nvx[ind_wall4])

    # Update positions
    nx = x + nvx * dt
    ny = y + nvy * dt

    #Look if any point has reached any of the exits
    ind_exit1 = np.where((nx>12) & (nx<17) & (ny>=28))
    
    if np.size(ind_exit1) > 0:
        ind_exit1 = ind_exit1[0][0] #Only one person can exit
        if exit1_time == 0:
            exit1_time = exit_cooldown
            nx[ind_exit1] = np.nan
            ny[ind_exit1] = np.nan
            vx[ind_exit1] = np.nan
            vy[ind_exit1] = np.nan
            print("Exit 1")
        else:
            exit1_time -=1

    ind_exit2 = np.where((nx>104) & (nx<109) & (ny>=40))

    if np.size(ind_exit2) > 0:
        ind_exit2 = ind_exit2[0][0]     
        if exit2_time == 0:
            exit2_time = exit_cooldown
            nx[ind_exit2] = np.nan
            ny[ind_exit2] = np.nan
            vx[ind_exit2] = np.nan
            vy[ind_exit2] = np.nan
            print("Exit 2")   
        else:
            exit2_time -=1

    ind_exit3 = np.where((ny>5) & (ny<10) & (nx>=140))  
    
    if np.size(ind_exit3) > 0:
        ind_exit3 = ind_exit3[0][0]
        if exit3_time == 0:
            exit3_time = exit_cooldown
            nx[ind_exit3] = np.nan
            ny[ind_exit3] = np.nan
            vx[ind_exit3] = np.nan
            vy[ind_exit3] = np.nan
            print("Exit 3")             
        else:
            exit3_time -=1

    for i in range(N2):

        #----------VÄGGAR--------#
        
        if nx[i] < 0:
            x[i] = -nx[i]
            vx[i] = -nvx[i]

        elif nx[i] > Lx:
            x[i] = Lx - (nx[i]-Lx)
            vx[i] = -nvx[i]
        else:
            x[i] = nx[i]
            vx[i] = nvx[i]

        if ny[i] < 0:
            y[i] = -ny[i]
            vy[i] = -nvy[i]

        elif ny[i] > Ly:
            y[i] = Ly - (ny[i]-Ly)
            vy[i] = -nvy[i]
        else:
            y[i] = ny[i]
            vy[i] = nvy[i]

        #------KLOSSAR----------#

        # BLOCK 5 - VÄNSTRE ÖVRE
        if nx[i] < 30 and ny[i] > 28:
            if nx[i] >=29:
                y[i] = ny[i]
                x[i] = 60-nx[i]
                vy[i] = nvy[i]
                vx[i] = -nvx[i]
            else:
                y[i] = 56-ny[i]
                x[i] = nx[i]
                vy[i] = -nvy[i]
                vx[i] = nvx[i]

        # BLOCK 4 - STORA KLOSSEN
        if nx[i] > 67 and nx[i] < 104 and ny[i] < 30:

            if ny[i] >= 28.5:
                y[i] = 60-ny[i]
                x[i] = nx[i]
                vy[i] = -nvy[i]
                vx[i] = nvx[i]
            elif nx[i] < 82:
                x[i] = 134-nx[i]
                y[i] = ny[i]
                vy[i] = nvy[i]
                vx[i] = -nvx[i] 
            else:
                x[i] = 208-nx[i]
                y[i] = ny[i]
                vy[i] = nvy[i]
                vx[i] = -nvx[i] 

        # BLOCK 2 - STÅENDE REKTANGEL
        if nx[i] > 44 and nx[i] < 58 and ny[i] < 20:

            if ny[i] >=19:
                y[i] = 40-ny[i]
                x[i] = nx[i]
                vy[i] = -nvy[i]
                vx[i] = nvx[i]

            elif nx[i] < 55:
                x[i] = 88-nx[i]
                y[i] = ny[i]
                vy[i] = nvy[i]
                vx[i] = -nvx[i]
            else:
                x[i] = 116-nx[i]
                y[i] = ny[i]
                vy[i] = nvy[i]
                vx[i] = -nvx[i]      

        # BLOCK 1 - NEDRE VÄNSTRA HÖRNET
        if nx[i] < 44 and ny[i] < 9:
            y[i] = 18-ny[i]
            x[i] = nx[i]
            vy[i] = -nvy[i]
            vx[i] = -nvx[i]

        # BLOCK 3 - LILLA KVADRATEN
        if nx[i] > 58 and nx[i] < 67 and ny[i] < 6:

            y[i] = 12-ny[i]
            x[i] = nx[i]

            vy[i] = -nvy[i]
            vx[i] = -nvx[i]
    #Break when all particles have evacuated        
    if np.all(np.isnan(x)):
        break 

print(f"The evacuation time was {t * dt:.2f} seconds")
# Points to visualize the boundaries
x1 = [0,0,0,12]
x2 = [17,30,30,30,30,104]
x3 = [109,140,140,140]
x4 = [140,140,140,104,104,104,104,67,67,67,67,58,58,58,58,44,44,44,44,0]
y1 = [9,28,28,28]
y2 = [28,28,28,40,40,40]
y3 = [40,40,40,10]
y4 = [5,0,0,0,0,30,30,30,30,6,6,6,6,20,20,20,20,9,9,9]

# Animation of trajectories
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-1, Lx + 1)
ax.set_ylim(-1, Ly + 1)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Plot boundaries in animation
ax.plot(x1, y1, color='black', lw=3)
ax.plot(x2, y2, color='black', lw=3)
ax.plot(x3, y3, color='black', lw=3)
ax.plot(x4, y4, color='black', lw=3)

# Initialize particle markers
particles, = ax.plot([], [], 'bo', markersize=8)  # 'bo' means blue circles

# Initialization function
def init():
    particles.set_data([], [])
    return particles,

def update(frame):
    x = x_path[frame, :]
    y = y_path[frame, :]

    # Exclude NaN values
    valid = ~np.isnan(x) & ~np.isnan(y)
    particles.set_data(x[valid], y[valid])
    return particles,


# Create the animation
ani = FuncAnimation(fig, update, frames=range(0, t, 10), init_func=init, blit=False, interval=50)
# Show the animation
plt.show()