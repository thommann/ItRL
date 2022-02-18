import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MountainCar:
    def __init__(self):
        """ Create a new mountain car object.
        
        It is possible to pass the parameter of the simulation.
        @param mass: the mass of the car (default 0.2) 
        @param friction:  the friction in Newton (default 0.3)
        @param delta_t: the time step in seconds (default 0.1)
        """
        
        self.alpha=0.001
        self.beta=0.0025
        self.eta=1
        self.position_max=-0.4
        self.position_list = list()
        

    def reset(self):
        """ It reset the car to an initial position [-1.2, 0.5]
        
        @param exploring_starts: if True a random position is taken
        @param initial_position: the initial position of the car (requires exploring_starts=False)
        @return: it returns the initial position of the car and the velocity
        """
        initial_position = np.random.uniform(-0.6,-0.4)
       
        self.position_list = []  # clear the list
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.position_list.append(initial_position)
        return [self.position_t, self.velocity_t]

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2]
         where 0=move_left, 1=do_not_move, 2=move_right
        @return: (postion_t1, velocity_t1), reward, done
         where reward is always negative but when the goal is reached
         done is True when the goal is reached
        """
        if(action >= 3):
            raise ValueError("[MOUNTAIN CAR][ERROR] The action value "
                             + str(action) + " is out of range.")
        done = False
        reward = -1
        action_list = [-1, 0, +1]
        action_t = action_list[action]
        
        velocity_t1=self.velocity_t+self.alpha*action_t-self.beta*np.cos(3*self.position_t)
        velocity_t1=np.min([velocity_t1,0.07])
        velocity_t1=np.max([velocity_t1,-0.07])
                            
        position_t1 = self.position_t + velocity_t1
        position_t1=np.min([position_t1,0.5])
        position_t1=np.max([position_t1,-1.2])
        
        # Check the limit condition (car outside frame)
        if position_t1 <= -1.2:
            velocity_t1 = 0
        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t= velocity_t1
        self.position_list.append(position_t1)
        # Reward and done when the car reaches the goal
       
        
        if position_t1 >= 0.5:
            reward = +1.0
            done = True
        # Return state_t1, reward, done
        return [position_t1, velocity_t1], reward, done

    
    def Create_Animation(self,X):
    
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.xlim=(-1.2, 0.5)
        ax.ylim=(-1.1, 1.1)
        particles, = ax.plot([], [], 'ro', ms=13)
    
        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
        y_sin = np.sin(3 * x_sin)
        ax.plot(x_sin, y_sin,'b')
        ax.plot(0.5,np.sin(3*0.5),'g^',ms=20)

        # initialization function: plot the background of each frame
        def init():
            particles.set_data([], [])
            return particles
    
        # animation function.  This is called sequentially
        def animate(i):
               
            particles.set_data(X[i],np.sin(3*X[i]))
    
            return particles
    
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(X)[0], interval=1, blit=True)
        #anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    
        return anim
        

