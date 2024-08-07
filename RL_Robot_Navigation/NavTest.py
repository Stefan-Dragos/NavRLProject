
import pygame
import math

pygame.init()
clock = pygame.time.Clock()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))

#---------------------Surfaces and Shapes-----------------------

ROBOT_WIDTH = 40
ROBOT_HEIGHT = 60

robotSurf = pygame.Surface((ROBOT_WIDTH,ROBOT_HEIGHT))
robotSurf.fill((180,255,0))
robotSurf.set_colorkey((0,0,0))

robotFront = pygame.Surface((ROBOT_WIDTH / 2, ROBOT_HEIGHT / 4))
robotFront.fill((255,0,0))
robotFront.set_colorkey((0,0,0))

robotWheel = pygame.Surface((ROBOT_WIDTH / 8, ROBOT_HEIGHT / 4))
robotWheel.fill((169,169,169))
robotWheel.set_colorkey((0,0,0))

robotSurf.blit(robotFront, (ROBOT_WIDTH / 4, ROBOT_HEIGHT / 8))
robotSurf.blit(robotWheel, (0, ROBOT_HEIGHT / 2))
robotSurf.blit(robotWheel, (ROBOT_WIDTH * (7/8), ROBOT_HEIGHT / 2))

image = robotSurf.copy()
image.set_colorkey((0,0,0))

robotRect = image.get_rect()
robotRect.center = [400,400]

#----------------------Variables--------------------------------

wheelOffset = ROBOT_WIDTH / 16

heading = 0  #relative to x axis
xPos = 0
yPos = 0  

forwardVel = 0
angularVel = 0

rightVel = 0
leftVel = 0
POWER_VEL_MULTIPLIER = 5


#--------------------Run Loop---------------------------
run = True
while run:

    clock.tick(30)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            run = False

    screen.fill((0,0,0))

#--------------------------------------------------------------

    key = pygame.key.get_pressed()

    if key[pygame.K_a]:
        leftVel = 1
    elif key[pygame.K_z]:
        leftVel = -1
    else:
        leftVel = 0

    if key[pygame.K_d]:
        rightVel = 1
    elif key[pygame.K_c]:
        rightVel = -1
    else:
        rightVel = 0

    leftVel = leftVel * POWER_VEL_MULTIPLIER
    rightVel = rightVel * POWER_VEL_MULTIPLIER


    forwardVel = (rightVel + leftVel) / 2
    angularVel = (rightVel - leftVel) / (2 * wheelOffset)

    #define rotation
    heading = (heading + angularVel) % 360

    deltaX = forwardVel * math.cos(math.radians(heading + 90))     #add 90 because we start facing y axis and heading is 0 degrees
    deltaY = forwardVel * math.sin(math.radians(heading + 90))
    
    xPos = xPos + deltaX
    yPos = yPos + deltaY
    robotRect.move_ip(deltaX, -1 * deltaY)

    print(f"xPos  {xPos}  |    yPos {yPos}")
    print()
    print(f"Improved: Xpos {robotRect.centerx}  |   yPos {robotRect.centery}")
    
    old_center = robotRect.center   #save old center
    new_image = pygame.transform.rotate(robotSurf , heading)  #make the rotation
    robotRect = new_image.get_rect()     #get new image rect
    robotRect.center = old_center    #translate back to original position

    #pygame.draw.rect(screen, (180,180,50), robotRect)  
    pygame.draw.circle(screen, (200,200, 200), [400,400], 15)

    hazard_positions = [[50,50,0],[200,40,0],[500,375,0]]

    for hazard in hazard_positions:
            pygame.draw.circle(screen, (255,0,40), hazard[0:2], 15)
    
    #-----------------------------------------------------------------
    
    screen.blit(new_image, robotRect)    #draw the rotated image on screen

    pygame.display.flip()  #updates the whole screen, unlike display.update which can update specific portions

pygame.quit()