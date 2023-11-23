'''
Two wheel dynamic model running in x,y coordinate using MPC Model and PID Controller 

'''

import numpy as np
import matplotlib.pyplot as plt
import support_mpc as sfc
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

# create an object for the support function
support = sfc.MPCsupport()
constants = support.constants

# Load the constant values from support file
Ts = constants['Ts']           # 10 second run time
outputs = constants['outputs'] # number of outputs (psi, Y)
hz = constants['hz'] # number of group
x_dot = constants['x_dot']  # constant horizontal speed
time_length = constants['time_length']

# Generate the reference point (from planner)
t = np.arange(0,time_length + Ts, Ts)    # from 0 to 10 with 0.1 second sample time
r = constants['r']   ## not neccessary
f = constants['f']   ## not neccesary

# take the planner input
psi_ref,X_ref,Y_ref = support.Trajectory_generator(t,r,f)  
sim_length = len(t) ## number of iteration
refSignals = np.zeros(len(X_ref)*outputs)


# Build up the reference point with following
# refsignal = [psi_ref_0,Y_ref_0, psi_ref_1, Y_ref_1 .......]
k = 0
for i in range(0,len(refSignals),outputs):
    refSignals[i] = psi_ref[k]
    refSignals[i+1] = Y_ref[k]
    k = k + 1

# setup the initial states
y_dot = 0.
psi = 0.
psi_dot = 0.
Y = Y_ref[0] + 10

states = np.array([y_dot,psi,psi_dot,Y])
statesTotal = np.zeros((len(t),len(states)))  # storring all states in this big matrix
statesTotal[0][0:len(states)] = states        # putt the inital state in Total
psi_opt_total = np.zeros((len(t),hz))         
Y_opt_total = np.zeros((len(t),hz))

# Load the initial input
U1 = 0 #input at t = -1 s(steering wheel angle in rad)
UTotal = np.zeros(len(t))
UTotal[0] = U1

# To extract psi_opt from predicted x_aug_opt
C_psi_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(1,hz+1):
    C_psi_opt[i-1][i+4*(i-1)]=1

# To extract Y_opt from predicted x_aug_opt
C_Y_opt=np.zeros((hz,(len(states)+np.size(U1))*hz))
for i in range(3,hz+3):
    C_Y_opt[i-3][i+4*(i-3)]=1

# generate the state space 
Ad,Bd,Cd,Dd = support.state_space()



# generate the mpc optimization terms
Hdb,Fdbt,Cdb,Adc=support.mpc_simplification(Ad,Bd,Cd,Dd,hz)


# initiate the controller - simulation loops
k = 0
for i in range(0,sim_length-1):
    
    # Generate the augmented current state and the reference vector
    x_aug_t = np.transpose([np.concatenate((states,[U1]),axis=0)])

    # From signal vector, only extract the reference value from your current sample + Ts
    # Example: T = 2 second, hz = 15 samples, so 
    # r = [psi_ref_2.1, Y_ref_2.1......psi_ref_3.5, Y_ref_3.5]
    k = k + outputs
    if k+outputs*hz <= len(refSignals):
        r = refSignals[k:k+outputs*hz]
    else:
        r = refSignals[k:len(refSignals)]
        hz = hz -1

    if hz < constants['hz']:
        # Generate the augmented current state and reference vector
        Hdb,Fdbt,Cdb,Adc = support.mpc_simplification(Ad,Bd,Cd,Dd,hz)

    ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)
    du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))
    x_aug_opt=np.matmul(Cdb,du)+np.matmul(Adc,x_aug_t)
    psi_opt=np.matmul(C_psi_opt[0:hz,0:(len(states)+np.size(U1))*hz],x_aug_opt)
    Y_opt=np.matmul(C_Y_opt[0:hz,0:(len(states)+np.size(U1))*hz],x_aug_opt)  

    psi_opt = np.transpose((psi_opt))[0]
    psi_opt_total[i+1][0:hz]=psi_opt
    Y_opt = np.transpose((Y_opt))[0]
    Y_opt_total[i+1][0:hz] = Y_opt

    U1 = U1 + du[0][0]

    '''PID Controller'''
    Pid_Switch = constants['PID_Switch']

    if Pid_Switch == 1:
        if i == 0:
            e_int_pid_yaw = 0
            e_int_pid_Y = 0
        if i > 0:
            e_pid_yaw_iml = psi_ref[i-1] - old_states[1]
            e_pid_yaw_i = psi_ref[i]-states[1]
            e_dot_pid_yaw = (e_pid_yaw_i - e_pid_yaw_iml)/Ts
            e_int_pid_yaw = e_int_pid_yaw + (e_pid_yaw_iml + e_pid_yaw_i)/2*Ts
            Kp_yaw=constants['Kp_yaw']
            Kd_yaw=constants['Kd_yaw']
            Ki_yaw=constants['Ki_yaw']
            U1_yaw = Kp_yaw * e_pid_yaw_i + Kd_yaw*e_dot_pid_yaw + Ki_yaw*e_int_pid_yaw

            e_pid_Y_iml = Y_ref[i-1] - old_states[3]
            e_pid_Y_i = Y_ref[i] - states[3]
            e_dot_pid_Y = (e_pid_Y_i - e_pid_Y_iml)/Ts
            e_int_pid_Y = e_int_pid_Y + (e_pid_Y_iml + e_pid_Y_i)/2*Ts
            Kp_Y = constants['Kp_Y']
            Kd_Y = constants['Kd_Y']
            Ki_Y = constants['Ki_Y']
            U1_Y = Kp_Y * e_pid_Y_i + Kd_Y * e_dot_pid_Y + Ki_Y*e_int_pid_Y

            U1 = U1_yaw + U1_Y

        old_states = states

        '''PID End'''
    if U1<-np.pi/6:
        U1 = -np.pi/6
    
    elif U1 > np.pi/6:
        U1 = np.pi/6
    else:
        U1 = U1

    # Keep track of the input
    UTotal[i+1] = U1

    #compute the new states
    states = support.open_loop_new_state(states,U1)
    statesTotal[i+1][0:len(states)] = states

################################ ANIMATION LOOP ###############################
# print(Y_opt_total)
# print(statesTotal)
# print(X_ref)
frame_amount=int(time_length/Ts)
lf=constants['lf']
lr=constants['lr']
# print(frame_amount)
def update_plot(num):

    hz = constants['hz'] # horizon prediction period

    car_1.set_data([X_ref[num]-lr*np.cos(statesTotal[num,1]),X_ref[num]+lf*np.cos(statesTotal[num,1])],
        [statesTotal[num,3]-lr*np.sin(statesTotal[num,1]),statesTotal[num,3]+lf*np.sin(statesTotal[num,1])])

    car_1_body.set_data([-lr*np.cos(statesTotal[num,1]),lf*np.cos(statesTotal[num,1])],
        [-lr*np.sin(statesTotal[num,1]),lf*np.sin(statesTotal[num,1])])

    car_1_body_extension.set_data([0,(lf+40)*np.cos(statesTotal[num,1])],
        [0,(lf+40)*np.sin(statesTotal[num,1])])

    car_1_back_wheel.set_data([-(lr+0.5)*np.cos(statesTotal[num,1]),-(lr-0.5)*np.cos(statesTotal[num,1])],
        [-(lr+0.5)*np.sin(statesTotal[num,1]),-(lr-0.5)*np.sin(statesTotal[num,1])])

    car_1_front_wheel.set_data([lf*np.cos(statesTotal[num,1])-0.5*np.cos(statesTotal[num,1]+UTotal[num]),lf*np.cos(statesTotal[num,1])+0.5*np.cos(statesTotal[num,1]+UTotal[num])],
        [lf*np.sin(statesTotal[num,1])-0.5*np.sin(statesTotal[num,1]+UTotal[num]),lf*np.sin(statesTotal[num,1])+0.5*np.sin(statesTotal[num,1]+UTotal[num])])

    car_1_front_wheel_extension.set_data([lf*np.cos(statesTotal[num,1]),lf*np.cos(statesTotal[num,1])+(0.5+40)*np.cos(statesTotal[num,1]+UTotal[num])],
        [lf*np.sin(statesTotal[num,1]),lf*np.sin(statesTotal[num,1])+(0.5+40)*np.sin(statesTotal[num,1]+UTotal[num])])

    yaw_angle_text.set_text(str(round(statesTotal[num,1],2))+' rad')
    steer_angle.set_text(str(round(UTotal[num],2))+' rad')

    steering_wheel.set_data(t[0:num],UTotal[0:num])
    yaw_angle.set_data(t[0:num],statesTotal[0:num,1])
    Y_position.set_data(t[0:num],statesTotal[0:num,3])

    if num+hz>len(t):
        hz=len(t)-num
    if Pid_Switch!=1 and num!=0:
        Y_predicted.set_data(t[num:num+hz],Y_opt_total[num][0:hz])
        psi_predicted.set_data(t[num:num+hz],psi_opt_total[num][0:hz])
        car_predicted.set_data(X_ref[num:num+hz],Y_opt_total[num][0:hz])
    car_determined.set_data(X_ref[0:num],statesTotal[0:num,3])

    if Pid_Switch!=1:
        return car_1, car_1_body, car_1_body_extension,\
        car_1_back_wheel, car_1_front_wheel, car_1_front_wheel_extension,\
        yaw_angle_text, steer_angle, steering_wheel,\
        yaw_angle, Y_position, car_determined, Y_predicted, psi_predicted, car_predicted
    else:
        return car_1, car_1_body, car_1_body_extension,\
        car_1_back_wheel, car_1_front_wheel, car_1_front_wheel_extension,\
        yaw_angle_text, steer_angle, steering_wheel,yaw_angle, Y_position, car_determined

# Set up your figure properties
fig_x=16
fig_y=9
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
n=3
m=3
gs=gridspec.GridSpec(n,m)

# Car motion

# Create an object for the motorcycle
ax0=fig.add_subplot(gs[0,:],facecolor=(0.9,0.9,0.9))

# Plot the reference trajectory
ref_trajectory=ax0.plot(X_ref,Y_ref,'b',linewidth=1)

# Plot the lanes
lane_width=constants['lane_width']
lane_1,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2,lane_width/2],'k',linewidth=0.2)
lane_2,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2,-lane_width/2],'k',linewidth=0.2)

lane_3,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2+lane_width,lane_width/2+lane_width],'k',linewidth=0.2)
lane_4,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2-lane_width,-lane_width/2-lane_width],'k',linewidth=0.2)

lane_5,=ax0.plot([X_ref[0],X_ref[frame_amount]],[lane_width/2+2*lane_width,lane_width/2+2*lane_width],'k',linewidth=3)
lane_6,=ax0.plot([X_ref[0],X_ref[frame_amount]],[-lane_width/2-2*lane_width,-lane_width/2-2*lane_width],'k',linewidth=3)

# Draw a motorcycle
car_1,=ax0.plot([],[],'k',linewidth=3)
car_predicted,=ax0.plot([],[],'-m',linewidth=1)
car_determined,=ax0.plot([],[],'-r',linewidth=1)

# Copyright
# copyright=ax0.text(0,20,'© Mark Misin Engineering',size=15)

# Establish the right (x,y) dimensions
plt.xlim(X_ref[0],X_ref[frame_amount])
plt.ylim(-X_ref[frame_amount]/(n*(fig_x/fig_y)*2),X_ref[frame_amount]/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]',fontsize=15)


# Create an object for the motorcycle (zoomed)
ax1=fig.add_subplot(gs[1,:],facecolor=(0.9,0.9,0.9))
bbox_props_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw='1')
bbox_props_steer_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw='1')

neutral_line=ax1.plot([-50,50],[0,0],'k',linewidth=1)
car_1_body,=ax1.plot([],[],'k',linewidth=3)
car_1_body_extension,=ax1.plot([],[],'--k',linewidth=1)
car_1_back_wheel,=ax1.plot([],[],'r',linewidth=4)
car_1_front_wheel,=ax1.plot([],[],'r',linewidth=4)
car_1_front_wheel_extension,=ax1.plot([],[],'--r',linewidth=1)

n1_start=-5
n1_finish=30
plt.xlim(n1_start,n1_finish)
plt.ylim(-(n1_finish-n1_start)/(n*(fig_x/fig_y)*2),(n1_finish-n1_start)/(n*(fig_x/fig_y)*2))
plt.ylabel('Y-distance [m]',fontsize=15)
yaw_angle_text=ax1.text(25,2,'',size='20',color='k',bbox=bbox_props_angle)
steer_angle=ax1.text(25,-2.5,'',size='20',color='r',bbox=bbox_props_steer_angle)

# Create the function for the steering wheel
ax2=fig.add_subplot(gs[2,0],facecolor=(0.9,0.9,0.9))
steering_wheel,=ax2.plot([],[],'-r',linewidth=1,label='steering angle [rad]')
plt.xlim(0,t[-1])
plt.ylim(np.min(UTotal)-0.1,np.max(UTotal)+0.1)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Create the function for the yaw angle
ax3=fig.add_subplot(gs[2,1],facecolor=(0.9,0.9,0.9))
yaw_angle_reference=ax3.plot(t,psi_ref,'-b',linewidth=1,label='yaw reference [rad]')
yaw_angle,=ax3.plot([],[],'-r',linewidth=1,label='yaw angle [rad]')
if Pid_Switch!=1:
    psi_predicted,=ax3.plot([],[],'-m',linewidth=3,label='psi - predicted [rad]')
plt.xlim(0,t[-1])
plt.ylim(np.min(statesTotal[:,1])-0.1,np.max(statesTotal[:,1])+0.1)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

# Create the function for the Y-position
ax4=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
Y_position_reference=ax4.plot(t,Y_ref,'-b',linewidth=1,label='Y - reference [m]')
Y_position,=ax4.plot([],[],'-r',linewidth=1,label='Y - position [m]')
if Pid_Switch!=1:
    Y_predicted,=ax4.plot([],[],'-m',linewidth=3,label='Y - predicted [m]')
plt.xlim(0,t[-1])
plt.ylim(np.min(statesTotal[:,3])-2,np.max(statesTotal[:,3])+2)
plt.xlabel('time [s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')


car_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)
plt.show()

# # Matplotlib 3.3.3 needed - comment out plt.show()
# Writer=animation.writers['ffmpeg']
# writer=Writer(fps=30,metadata={'artist': 'Me'},bitrate=1800)
# car_ani.save('car1.mp4',writer)

##################### END OF THE ANIMATION ############################



# Plot the world
plt.plot(X_ref,Y_ref,'b',linewidth=2,label='The trajectory')
plt.plot(X_ref,statesTotal[:,3],'--r',linewidth=2,label='Car position')
plt.xlabel('x-position [m]',fontsize=15)
plt.ylabel('y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.ylim(-X_ref[-1]/2,X_ref[-1]/2) # Scale roads (x & y sizes should be the same to get a realistic picture of the situation)
plt.show()


# Plot the the input delta(t) and the outputs: psi(t) and Y(t)
plt.subplot(3,1,1)
plt.plot(t,UTotal[:],'r',linewidth=2,label='steering wheel angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('steering wheel angle [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,psi_ref,'b',linewidth=2,label='Yaw_ref angle')
plt.plot(t,statesTotal[:,1],'--r',linewidth=2,label='Car yaw angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('psi_ref-position [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,Y_ref,'b',linewidth=2,label='Y_ref position')
plt.plot(t,statesTotal[:,3],'--r',linewidth=2,label='Car Y position')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')
plt.show()

''' Also treat the case when Ki-s are 0'''




