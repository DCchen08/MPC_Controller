'''
This is the support file to MPC.py


'''


import numpy as np
import matplotlib as plt

class MPCsupport:

    '''this is the function that will implement in MPC.py file '''
    def __init__(self):
        ''' Load the constants that do not change '''

        #Constants
        m = 1500
        Iz = 3000
        Caf = 19000
        Car = 33000
        lf = 2
        lr = 3
        Ts = 0.02

        # Parameters for the lane change: [psi_ref 0;0 Y_ref]
        # Higher psi reduces the overshoot

        # Matrix weights for the cost function (They must be diagonal)
        Q=np.matrix('1 0;0 1') # weights for outputs (all samples, except the last one)
        S=np.matrix('1 0;0 1') # weights for the final horizon period outputs
        R=np.matrix('1') # weights for inputs (only 1 input in our case)

        outputs = 2
        hz = 20
        x_dot=20 # car's longitudinal velocity
        lane_width=7 # [m]
        nr_lanes=5 # 6 lanes [m]

        r = 4    # r = amplitude of sinusoidal function
        f = 0.01 # f is the frequency f = 0.01 * np.random.randint(2)+0.01
        time_length = 10


        ## PID Controller
        PID_Switch = 0

        Kp_yaw = 1
        Kd_yaw = 3
        Ki_yaw = 5

        Kp_Y=7
        Kd_Y=3
        Ki_Y=5


        self.constants = {'m':m,
                          'Iz':Iz,
                          'Caf':Caf, 
                          'Car':Car,
                          'lf':lf, 
                          'lr':lr,  
                          'Ts':Ts, 
                          'Q':Q,
                          'S':S,
                          'R':R,
                          'outputs':outputs,
                          'hz':hz,
                          'x_dot':x_dot,
                          'lane_width':lane_width,
                          'nr_lane':nr_lanes,
                          'time_length':time_length,
                          'PID_Switch':PID_Switch,
                          'Kp_yaw':Kp_yaw,
                          'Kd_yaw':Kd_yaw,
                          'Ki_yaw':Ki_yaw,
                          'Kp_Y':Kp_Y,
                          'Kd_Y':Kd_Y,
                          'Ki_Y':Ki_Y,
                          'r':r,
                          'f':f}
        
        return None
        
    def Trajectory_generator(self,t,r,f):
        '''This method creates the trajectory for a car (Reference point)'''
        Ts = self.constants['Ts']
        x_dot = self.constants['x_dot']

        # Define x length
        x = np.linspace(0,x_dot*t[-1], num=len(t))
        
        # Define the y(trajector)
        y=9*np.tanh(t-t[-1]/2)

        # Vector of x and y changes
        dx = x[1:len(x)] - x[0:len(x)-1]
        dy = y[1:len(y)] - y[0:len(y)-1]

        # Define the reference angle(PSI) 
        ''' psi = arctan(dy/dx)'''
        psi = np.zeros(len(x))   #initial the vector
        psiInt = psi
        psi[0] = np.arctan2(dy[0],dx[0])  # initial value of psi
        psi[1:len(psi)] = np.arctan2(dy[0:len(dy)], dx[0:len(x)])  # rest of those

        ''' We want yaw angle to keep track the amount of relations '''
        dpsi = psi[1:len(psi)] - psi[0:len(psi) - 1]
        psiInt[0] = psi[0]

        for i in range(1,len(psiInt)):
            if dpsi[i-1] < -np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        return  psiInt,x,y
    

    def state_space(self):
        '''This function form the state matrix and transforms them in the discrete form'''

        # getting the constants from constructor
        m = self.constants['m']
        Iz = self.constants['Iz']
        Caf = self.constants['Caf']
        Car = self.constants['Car']
        lf = self.constants['lf']
        lr = self.constants['lr']
        Ts = self.constants['Ts']
        x_dot = self.constants['x_dot']

        # Get the state space matrices for the control
        A1 = -(2*Caf + 2*Car)/(m*x_dot)
        A2 = -x_dot - (2*Caf*lf - 2*Car*lr)/(m*x_dot)
        A3 = -(2*lf*Caf - 2*lr*Car)/(Iz*x_dot)
        A4 = -(2*lf**2*Caf + 2*lr**2*Car)/(Iz*x_dot)
        

        A = np.array([[A1,0,A2,0],[0,0,1,0],[A3,0,A4,0],[1,x_dot,0,0]])
        B = np.array([[2*Caf/m],[0],[2*lf*Caf/Iz],[0]])
        C = np.array([[0,1,0,0],[0,0,0,1]])
        D = 0

        # Discrete the system with forward
        Ad = np.identity(np.size(A,1)) + Ts*A
        Bd = Ts * B
        Cd = C
        Dd = D

        return Ad,Bd,Cd,Dd
    

    def mpc_simplification(self,Ad,Bd,Cd,Dd,hz):
        '''this function is to create the matrix for MPC Controller(Model Predictive Control)'''

        A_aug = np.concatenate((Ad,Bd),axis=1)   ## This is for creating the matrix A_aug = [Ad Bd]
        temp1 = np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2 = np.identity(np.size(Bd,1))
        temp  = np.concatenate((temp1,temp2),axis=1)

        # forming the A_aug, B_aug, C_aug, D_aug matrix
        A_aug = np.concatenate((A_aug,temp),axis = 0)
        B_aug = np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug = np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug = Dd

        # forming the optimizing stage
        Q = self.constants['Q']
        S = self.constants['S']
        R = self.constants['R']

        CQC = np.matmul(np.transpose(C_aug),Q)
        CQC = np.matmul(CQC,C_aug)


        CSC = np.matmul(np.transpose(C_aug),S)
        CSC = np.matmul(CSC,C_aug)

        QC = np.matmul(Q,C_aug)
        SC = np.matmul(S,C_aug)

        Qdb = np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))   # Q double bar
        Tdb = np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb = np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb = np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc = np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
             if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
             else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

             Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

             for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

             Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc
    


    def open_loop_new_state(self,states,U1):
        '''This function computes the new state vector for one simple time later'''

        # get the constant
        m = self.constants['m']
        Iz = self.constants['Iz']
        Caf = self.constants['Caf']
        Car = self.constants['Car']
        lf = self.constants['lf']
        lr = self.constants['lr']
        Ts = self.constants['Ts']
        x_dot = self.constants['x_dot']

        current_state = states
        new_states = current_state
        y_dot = current_state[0]
        psi = current_state[1]
        psi_dot = current_state[2]
        Y = current_state[3]

        sub_loop = 30    # Cutting Ts into 30 piece
        for i in range(0,sub_loop):
            # compute the derivative of states to calculate the next state
            y_dot_dot=-(2*Caf+2*Car)/(m*x_dot)*y_dot+(-x_dot-(2*Caf*lf-2*Car*lr)/(m*x_dot))*psi_dot+2*Caf/m*U1
            psi_dot=psi_dot
            psi_dot_dot=-(2*lf*Caf-2*lr*Car)/(Iz*x_dot)*y_dot-(2*lf**2*Caf+2*lr**2*Car)/(Iz*x_dot)*psi_dot+2*lf*Caf/Iz*U1
            Y_dot=np.sin(psi)*x_dot+np.cos(psi)*y_dot

            # Update the new states value
            y_dot=y_dot+y_dot_dot*Ts/sub_loop
            psi=psi+psi_dot*Ts/sub_loop
            psi_dot=psi_dot+psi_dot_dot*Ts/sub_loop
            Y=Y+Y_dot*Ts/sub_loop

        
        # Take the last states
        new_states[0] = y_dot
        new_states[1] = psi
        new_states[2] = psi_dot
        new_states[3] = Y


        return new_states








