import retro
import numpy as np
#import torch
#import torch.nn as nn
from q_learning_N import DQN

'''
class DQN(nn.Module):

    def __init__(self, observations, actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=(4, 4))
        self.layer2 = nn.Conv2d(32, 64, kernel_size=(4, 4))
        self.layer3 = nn.Linear(64, 8)

    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))  
        return self.layer3(x)
'''




color = np.array([224, 240, 74]).mean()

def preprocess_state(state):

    #crop and resize the image
    image = state[::2, ::2]

    #convert the image to greyscale
    image = image.mean(axis=2)

    #improve image contrast
    image[image==color] = 0

    #normalize the image
    image = (image - 128) / 128 - 1
    
    image = np.expand_dims(image.reshape(112, 120, 1), axis=0)

    return image

def action_to_list(a):
    actions=[0,0,0,0,0,0,0,0,0]
    actions[a]=1
    return actions

def main():
    num_episodes = 500
    num_timesteps = 20000
    batch_size = 8
    num_screens = 4
    state_size = (112, 120, 1)

    env = retro.make(game='ContraForce-Nes')
    action_size = env.action_space.n
   
    dqn = DQN(state_size, action_size)
    '''
    obs = env.reset()
    img = preprocess_state(obs)
    print(img.shape)
    

    while True:
        obs, rew, done, info = env.step([1,0,0,0,0,0,0,0,1])
        env.render()
        if done:
            print("over")
    env.close()
    '''

    done = False
    time_step = 0

    #for each episode
    for i in range(num_episodes):
        
        #set return to 0
        Return = 0
        
        #preprocess the game screen
        state = preprocess_state(env.reset())

        #for each step in the episode
        for t in range(num_timesteps):
            
            #render the environment
            env.render()
            
            #update the time step
            time_step += 1
            
            #update the target network
            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()
            
            #select the action
            action = dqn.epsilon_greedy(state)

            real_action=action_to_list(action)
            
            #perform the selected action
            next_state, reward, done, _ = env.step(real_action)
            
            #preprocess the next state
            next_state = preprocess_state(next_state)
            
            #store the transition information
            dqn.store_transistion(state, action, reward, next_state, done)
            
            #update current state to next state
            state = next_state
            
            #update the return
            Return += reward
            
            #if the episode is done then print the return
            if done:
                print('Episode: ',i, ',' 'Return', Return)
                break
                
            #if the number of transistions in the replay buffer is greater than batch size
            #then train the network
            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)


if __name__ == "__main__":
    main()
