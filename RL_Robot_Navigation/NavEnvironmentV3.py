import gymnasium as gym
from gymnasium import Env
import numpy as np
from numpy import random
import pygame
import math
import matplotlib.pyplot as plt

#TODO --> Send out 5 rays from the from of the car in 8 degree increments and return the distance intersected
class NavEnvV3(Env):

    metadata = {"render_modes": ["human"], "render_fps": 16}

    def __init__(self, hazards, power_multiplier, envSize, render_mode = None):
        #Sets maximum steps per episode, initializes current step to 0
        self.maxSteps = 300
        self.currentSteps = 0
        #Set number of hazards
        self.hazardNum = hazards

        #Set environment size
        if (envSize < 600):
            envSize = 600
        self.ENV_SIZE = envSize

        #total rewards to then plot
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

        self.TARGET_RADIUS = 25
        self.HAZARD_RADIUS = 20

        self.closeHazard = []
        self.closeHazardDistance = 0
        self.closeHazardAngle = 0

        self.closeHazard2 = []
        self.closeHazard2Distance = 0
        self.closeHazard2Angle = 0

        self.seed = None

        #Assigning environment render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        #init pygame objects for rendering
        if self.render_mode == "human":
            self._init_visual()

        #set spaces after all vars are initialized
        #setting observation space, one 3D position + heading vector per object (player, target, n hazards)
        self.observation_space = self.getObsSpace()
        #setting the action spaces, one 2D vector representing left and right motor powers from (-1 --> +1)
        self.action_space = gym.spaces.Box(
            low = -1,
            high = 1,
            shape = (2,),
            dtype = np.float16
        )
        

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
        self.angularVel = (self.rightVel - self.leftVel) / (2 * self.WHEEL_OFFSET)
        #update heading, needs to be int to satisfy obs space
        self.heading = (self.heading + self.angularVel) % 360
        #get change in x and y pos
        deltaX = self.forwardVel * math.cos(math.radians(self.heading + 90))     #add 90 because we start facing y axis and heading is 0 degrees
        deltaY = self.forwardVel * math.sin(math.radians(self.heading + 90))
        #update positions, need to be int to satisfy obs space
        self.xPos = np.clip(self.xPos + deltaX, 0, 600)
        self.yPos = np.clip(self.yPos + deltaY, 0, 600)

        #update agent position
        self.agent_position = [self.xPos, self.yPos, self.heading]
        #update closest hazard and dist
        self.closeHazard, self.closeHazardDistance, self.closeHazard2, self.closeHazard2Distance = self.closestHazard()
        #get the relative angle to the hazard1, hazard2
        self.closeHazardAngle = self.agentAngleTo(self.closeHazard)
        self.closeHazard2Angle = self.agentAngleTo(self.closeHazard2)
        #update observation
        self.current_observation = self.get_observation()

        #calculate distance to target for rewards
        distanceToTarget = math.hypot(self.target_position[0] - self.agent_position[0], 
                        self.target_position[1] - self.agent_position[1])
        
        #evaluate reward for facing the target
        angleRew = 0

        angleToTarget = self.agentAngleTo(self.target_position)
        if angleToTarget > 30 and angleToTarget <= 180:
            angleRew = -1 * angleToTarget / 360
        elif angleToTarget < 330 and angleToTarget > 180:
            angleRew = -1 * (angleToTarget - 150) / 360

        #evaluate rewards, don't add target angle reward when reaching the target
        if self.hazardContact():
            terminated = False
            reward = -25 + angleRew
            print("HAZARD CONTACT")
        elif distanceToTarget < self.TARGET_RADIUS:
            terminated = True
            reward = 10
            print("GOT TO TARGET")
        elif self.agent_position[0] > 590 or self.agent_position[0] < 10 or self.agent_position[1] > 595 or self.agent_position[1] < 5:
            reward = -1 * distanceToTarget / 100 + angleRew
        else:
            reward = -1 * distanceToTarget / 500 + angleRew #- self.currentSteps / 100  #negative reward proportional to distance to target
                                                                      
        self.render()

        self.TOTAL_REW.append(reward)

        return self.current_observation, reward, terminated, False, {}  #info has to be dict


    def reset(self, seed = None, options = None):
        #Reset superclass with input seed
        super().reset(seed=seed)

        #seeding the randomness to recreate observations
        self.seed = seed
        rng = random.RandomState(self.seed)

        #Reset current steps
        self.currentSteps = 0

        #randomize agent and target position
        #self.agent_position = [150,75,0] 
        #self.target_position = [365, 515, 0] 
        self.agent_position = [rng.randint(100,500), rng.randint(50,100), 0]
        self.target_position = [rng.randint(50,550), rng.randint(500,600), 0]

        self.xPos = self.agent_position[0]
        self.yPos = self.agent_position[1]
        self.heading = self.agent_position[2]

        #reset hazard position with one random pos
        #self.hazard_positions = [[140,275,0],[320,175,0],[200,400,0],[375,265,0],[335, 465, 0]]
        self.hazard_positions = [[rng.randint(10,590), rng.randint(110,490), 0]]

        #fill in the rest of hazard positions
        for i in range(self.hazardNum - 1):
            self.hazard_positions.append([rng.randint(10,590), rng.randint(110,490), 0])
        
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

        #draw line to all hazards
        for hazard in self.hazard_positions:
            pygame.draw.line(self.SCREEN, (255,0,255), (round(self.agent_position[0]), round(600 - self.agent_position[1])), (round(hazard[0]), round(600 - hazard[1])))
        
        #draw line to the target
        pygame.draw.line(self.SCREEN, (50,215,180), (round(self.agent_position[0]), round(600 - self.agent_position[1])), (round(self.target_position[0]), round(600 - self.target_position[1])), width=3)

        #update display
        pygame.display.flip()


    def get_observation(self):
        #get current observation until hazards
        obs = [self.agent_position[0], self.agent_position[1], self.agent_position[2], 
                self.forwardVel, self.angularVel, 
                self.agentDistanceTo(self.target_position), self.agentAngleTo(self.target_position), 
                ]
        #append all hazard distance and angles to end of obs
        for hazard in self.hazard_positions:
            obs.append(self.agentDistanceTo(hazard))
            obs.append(self.agentAngleTo(hazard))

        return obs


    def getObsSpace(self):

        #return the space: [agent X, agentY, agentHeading, agentLinearVel, agentAngularVel, targetDistance, targetAngle, h1Dist, h1Angle, h2Dist, h2Angle....h5Dist, h5Angle]
        space = gym.spaces.Box(
            low = np.array([0, 0, 0, -1 * self.POWER_VEL_MULTIPLIER, -1 * self.POWER_VEL_MULTIPLIER / self.WHEEL_OFFSET, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            high = np.array([600, 600, 360, self.POWER_VEL_MULTIPLIER, self.POWER_VEL_MULTIPLIER / self.WHEEL_OFFSET, 800, 360, 800, 360, 800, 360, 800, 360, 800, 360, 800, 360]),
            shape = (17,),
            dtype = np.float32
        )

        return space


    def hazardContact(self):
        for i, hazard in enumerate(self.hazard_positions):
            if self.agentDistanceTo(hazard) <= self.HAZARD_RADIUS:
                return True
            
        return False


    def closestHazard(self):
        #instantiate closest hazard and its distance
        closestDistance = 10000
        closestDistance2 = 20000
        closestHazard = []
        closestHazard2 = []

        for hazard in self.hazard_positions:
            dist = self.agentDistanceTo(hazard)
            if dist < closestDistance:
                closestDistance = dist
                closestHazard = hazard

        for hazard2 in self.hazard_positions:
            dist2 = self.agentDistanceTo(hazard2)
            if dist2 < closestDistance2 and dist2 > closestDistance:
                closestDistance2 = dist2
                closestHazard2 = hazard2

        return closestHazard, closestDistance, closestHazard2, closestDistance2
    

    def agentDistanceTo(self, point):
        return math.hypot(point[0] - self.agent_position[0], point[1] - self.agent_position[1])


    def agentAngleTo(self, point):
        return (math.degrees(
            math.atan2(point[1] - self.agent_position[1], point[0] - self.agent_position[0]))
              - (self.heading + 90)) % 360


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


    def lineCircleIntersection(self, circleCenter, radius, p1, p2):
        #init intersection list
        intersections = []

        #get rid of horizontal and vertical line cases
        if abs(p1[1] - p2[1]) < 0.01:
            p1[1] = p1[1] + 0.01
        if abs(p1[0] - p2[0]) < 0.01:
            p1[0] = p2[0] + 0.01

        #define slope
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        #make coordinates relative to circle center
        x1 = p1[0] - circleCenter[0]
        y1 = p1[1] - circleCenter[1]
        #define quadratic terms
        quadA = 1.0 + math.pow(slope, 2)
        quadB = (2.0 * slope * y1) - (2.0 * math.pow(slope,2) * x1)
        quadC = ((math.pow(slope, 2) * math.pow(x1, 2)) - (2.0 * y1 * slope * x1) + math.pow(y1, 2) - math.pow(radius, 2))

        try:
            #quadratic formula
            xRoot1 = (-quadB + math.sqrt(math.pow(quadB, 2) - 4 * quadA * quadC)) / (2.0 * quadA)
            yRoot1 = slope * (xRoot1 - x1) + y1

            #reverse the offset from making calculations relative to circle center
            xRoot1 += circleCenter[0]
            yRoot1 += circleCenter[1]
            #define line segment bounds
            minX = min(p1[0], p2[0])
            maxX = max(p1[0], p2[0])
            #make sure point is on line segment
            if xRoot1 > minX and xRoot1 < maxX:
                intersections.append([xRoot1, yRoot1])

            #get second root with quadratic formula
            xRoot2 = (-quadB - math.sqrt(math.pow(quadB, 2) - 4 * quadA * quadC)) / (2.0 * quadA)
            yRoot2 = slope * (xRoot2 - x1) + y1

            #reverse the offset from making calculations relative to circle center
            xRoot2 += circleCenter[0]
            yRoot2 += circleCenter[1]
            #line segment bounds
            if xRoot2 > minX and xRoot2 < maxX:
                intersections.append([xRoot2, yRoot2])
            
            
        except:
            pass
        
        return intersections
