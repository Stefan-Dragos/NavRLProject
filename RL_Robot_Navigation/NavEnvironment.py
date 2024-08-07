import gymnasium as gym
from gymnasium import Env
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pygame
import math

#TODO --> Make a secondary environment with different observation space:
#Agent Position (x,y,heading), Agent Velocity (dx/dt,dy/dt), Target Position, Position of closest hazard relative to agent (x,y)
#For pos of closest hazard, iterate through all hazards, get distances, and return difference in x and y pos relative to agent
class NavEnv(Env):

    metadata = {"render_modes": ["human"], "render_fps": 16}

    def __init__(self, hazards, power_multiplier, envSize, render_mode = None):
        #Sets maximum steps per episode, initializes current step to 0
        self.maxSteps = 200
        self.currentSteps = 0
        #Set number of hazards
        self.hazardNum = hazards

        #Set environment size
        if (envSize < 600):
            envSize = 600
        self.ENV_SIZE = envSize

        #setting observation space, one 3D position + heading vector per object (player, target, n hazards)
        self.observation_space = self.getObsSpace()
        #setting the action spaces, one 2D vector representing left and right motor powers from (-1 --> +1)
        self.action_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (2,),
            dtype = np.float16
        )
        #set total reward
        self.TOTAL_REW = []

        #Setting variables ---------------------------------------------------
        self.agent_position = [0,0,0]
        self.target_position = [600,600,0]
        self.hazard_positions = [100,100,0] * self.hazardNum

        self.ROBOT_WIDTH = 40.0
        self.ROBOT_HEIGHT = 60.0

        self.heading = 0  #relative to x axis
        self.xPos = 0
        self.yPos = 0

        self.forwardVel = 0
        self.angularVel = 0

        self.rightVel = 0
        self.leftVel = 0
        self.POWER_VEL_MULTIPLIER = power_multiplier
        self.WHEEL_OFFSET = self.ROBOT_WIDTH / 16

        self.TARGET_RADIUS = 50
        self.HAZARD_RADIUS = 10


        #Assigning environment render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        #init pygame objects for rendering
        if self.render_mode == "human":
            self._init_visual()
        

    def step(self, action):
        terminated = False
        reward = 0
        #update current steps and check for maximum
        self.currentSteps += 1

        if self.currentSteps > self.maxSteps:
            terminated = True

        #get motor powers from action
        self.leftVel = action[0] * self.POWER_VEL_MULTIPLIER
        self.rightVel = action[1] * self.POWER_VEL_MULTIPLIER
        #translate into linear and angular velocity
        self.forwardVel = (self.rightVel + self.leftVel) / 2
        self.angularVel = (self.rightVel - self.leftVel) / (2 * 3)
        #update heading, needs to be int to satisfy obs space
        self.heading = (self.heading + self.angularVel) % 360
        #get change in x and y pos
        deltaX = self.forwardVel * math.cos(math.radians(self.heading + 90))     #add 90 because we start facing y axis and heading is 0 degrees
        deltaY = self.forwardVel * math.sin(math.radians(self.heading + 90))
        #update positions, need to be int to satisfy obs space
        self.xPos = np.clip(self.xPos + deltaX, 0, 600)
        self.yPos = np.clip(self.yPos + deltaY, 0, 600)
        
        #update observation and agent pos
        self.current_observation = self.get_observation()
        #calculate distance to target
        distanceToTarget = math.hypot(self.target_position[0] - self.agent_position[0], 
                        self.target_position[1] - self.agent_position[1])

        #evaluate rewards
        if self.hazardContact():
            terminated = False
            reward = -10
            print("HAZARD CONTACT")
        elif distanceToTarget < self.TARGET_RADIUS:
            terminated = True
            reward = 10
            print("GOT TO TARGET")
        elif self.agent_position[0] > 590 or self.agent_position[0] < 10 or self.agent_position[1] > 595 or self.agent_position[1] < 5:
            reward = -1 * distanceToTarget / 100
            print("EDGES")
        else:
            reward = -1 * distanceToTarget / 500   #negative reward proportional to distance to target
                                                                      
        self.render()

        self.TOTAL_REW.append(reward)

        return self.current_observation, reward, terminated, False, {}  #info has to be dict


    def reset(self, seed = None):
        #Reset superclass with input seed
        super().reset(seed=seed)

        #Reset current steps
        self.currentSteps = 0

        #randomize agent and target position
        self.agent_position = [150,75,0] #[random.randint(100,500), random.randint(50,100), 0]
        self.target_position = [400, 565, 0]#[random.randint(50,550), random.randint(500,600), 0]
        self.xPos = self.agent_position[0]
        self.yPos = self.agent_position[1]
        self.heading = self.agent_position[2]

        #reset hazard position with one random pos
        self.hazard_positions = [[140,275,0],[320,175,0],[200,400,0],[375,265,0],[335, 465, 0]] #[[random.randint(10,590), random.randint(110,490), 0]]

        #fill in the rest of hazard positions
        #for i in range(self.hazardNum - 1):
            #self.hazard_positions.append([random.randint(10,590), random.randint(110,490), 0])
        
        #update render positions
        if self.render_mode == "human":
            self.robotRect.center = self.agent_position[0:2]

        #get new observation
        self.current_observation = self.get_observation()
        #no info
        info = {}

        return self.current_observation, info
    

    def render(self):
        #exit function if there is no rendering
        if self.render_mode is None:
            return
        
        pygame.event.get()
        
        #set framerate
        self.CLOCK.tick(self.metadata["render_fps"])
        #reset screen
        self.SCREEN.fill((0,0,0))
        
        #move agent
        self.robotRect.center = (round(self.agent_position[0]), round(600 - self.agent_position[1]))
        #draw agent to screen
        old_center = self.robotRect.center   #save old center
        new_image = pygame.transform.rotate(self.robotSurf , round(self.heading))  #make the rotation
        self.robotRect = new_image.get_rect()     #get new image rect
        self.robotRect.center = old_center    #translate back to original position
        self.SCREEN.blit(new_image, self.robotRect)    #draw the rotated image on screen

        #draw target
        pygame.draw.circle(self.SCREEN, (0,255,120), (round(self.target_position[0]), round(600 - self.target_position[1])), self.TARGET_RADIUS)
        #draw all hazards
        for hazard in self.hazard_positions:
            pygame.draw.circle(self.SCREEN, (255,0,40), (round(hazard[0]), round(600 - hazard[1])), self.HAZARD_RADIUS)

        #update display
        pygame.display.flip()


    def get_observation(self):
        #update agent position
        self.agent_position = [self.xPos, self.yPos, self.heading]
        #Setup observation with target and agent pos
        obs = [self.agent_position, self.target_position]
        #iterate through each hazard, append it to obs
        for i in range(len(self.hazard_positions)):
            obs.append(self.hazard_positions[i])

        return obs


    def getObsSpace(self):
        #sets the lower and higher ranges of 1) agent, and 2) target
        lowRanges = [[0,0,0],
                     [50,500,0]]
        
        highRanges = [[600,600,360],
                      [550, 600,0]]

        #extends low and high ranges to hazards
        for i in range(self.hazardNum):
            lowRanges.append([10,10,0])
            highRanges.append([590,590,0])

        #return the space
        space = gym.spaces.Box(
            low = np.array(lowRanges),
            high = np.array(highRanges),
            shape = (2 + self.hazardNum,3),
            dtype = np.float32
        )

        return space


    def hazardContact(self):
        for i, hazard in enumerate(self.hazard_positions):
            if math.hypot(hazard[0] - self.agent_position[0], hazard[1] - self.agent_position[1]) <= self.HAZARD_RADIUS:
                return True
            
        return False


    def closestHazard(self):
            #instantiate closest hazard and its distance
            closestDistance = 10000
            closestHazard = 0

            for hazard in self.hazard_positions:
                dist = math.hypot(hazard[0] - self.agent_position[0], hazard[1] - self.agent_position[1])
                if dist < closestDistance:
                    closestDistance = dist
                    closestHazard = hazard

            return closestHazard, closestDistance


    def _init_visual(self):
        #all pygame initialization here  (ONLY INIT IF RENDER MODE IS HUMAN):
        pygame.init()
        #get pygame clock
        self.CLOCK = pygame.time.Clock()
        #initalize the screen
        self.SCREEN = pygame.display.set_mode((600,600))
        #init all robot surfaces
        self.robotSurf = pygame.Surface((self.ROBOT_WIDTH,self.ROBOT_HEIGHT))
        self.robotSurf.fill((180,255,0))
        self.robotSurf.set_colorkey((0,0,0))

        self.robotFront = pygame.Surface((self.ROBOT_WIDTH / 2, self.ROBOT_HEIGHT / 4))
        self.robotFront.fill((255,0,0))
        self.robotFront.set_colorkey((0,0,0))

        self.robotWheel = pygame.Surface((self.ROBOT_WIDTH / 8, self.ROBOT_HEIGHT / 4))
        self.robotWheel.fill((169,169,169))
        self.robotWheel.set_colorkey((0,0,0))

        self.robotSurf.blit(self.robotFront, (self.ROBOT_WIDTH / 4, self.ROBOT_HEIGHT / 8))
        self.robotSurf.blit(self.robotWheel, (0, self.ROBOT_HEIGHT / 2))
        self.robotSurf.blit(self.robotWheel, (self.ROBOT_WIDTH * (7/8), self.ROBOT_HEIGHT / 2))

        self.robotImage = self.robotSurf.copy()
        self.robotImage.set_colorkey((0,0,0))

        self.robotRect = self.robotImage.get_rect()
        self.robotRect.center = (300,300)
        #target and hazards will be drawn directly to screen in render() method


    def showRewardPlot(self, totalSteps):

        plt.plot(self.TOTAL_REW)
        plt.xticks(np.arange(0, totalSteps, totalSteps / 10))
        plt.show()