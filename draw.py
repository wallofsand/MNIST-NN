import sys
import pygame
import ctypes
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize as re
from utils import *

# Pygame config
pygame.init()
fps = 300
fpsClock = pygame.time.Clock()
width, height = 1280, 960
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
font = pygame.font.SysFont('Arial', 20)
# Increase Dots Per inch so it looks sharper
ctypes.windll.shcore.SetProcessDpiAwareness(True)

# Variables
# Our Buttons will append themselves to this list
objects = []
# Initial color
drawColor = [255, 255, 255]
shadeColor = [150, 150, 150]
# Initial brush size
brushSize = 10
brushSizeSteps = 3
# Drawing Area Size
canvasSize = [244, 244]
displaySize = [28, 28]
# Button Variables.
buttonWidth = 120
buttonHeight = 35
# Canvas
canvas = pygame.Surface(canvasSize)
canvas.fill((0, 0, 0))
# Display the shrunken image
# display = pygame.Surface((28, 28))
# display.fill((255, 255, 255))

# Button Class
class Button():
    def __init__(self, x, y, width, height, buttonText='Button', onclickFunction=None, onePress=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.onePress = onePress
        self.fillColors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }
        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)
        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))
        self.alreadyPressed = False
        objects.append(self)

    def process(self):
        mousePos = pygame.mouse.get_pos()
        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])
            if pygame.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])
                if self.onePress:
                    self.onclickFunction()
                elif not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True
            else:
                self.alreadyPressed = False
        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)

# Handler Functions
# Changing the Color
def changeColor(color):
    global drawColor
    drawColor = color

# Changing the Brush Size
def changebrushSize(dir):
    global brushSize
    if dir == 'greater':
        brushSize += brushSizeSteps
    else:
        brushSize -= brushSizeSteps

# Save the surface to the Disk
def save():
    pygame.image.save(canvas, "canvas.png")

# Clear the canvas
def clear():
    canvas.fill((0, 0, 0))
    # display.fill((255, 255, 255))

# Pass the image through the network
def run(network):
    image = np.zeros((224,224))
    for col in range(224):
        for row in range(224):
            image[row,col] = canvas.get_at((col,row))[0]
    prcimg = lib_downsample(image)
    output = network.forward_pass(prcimg)
    print('Testing image:')
    for pair in enumerate(output):
        print(pair)
    print('This looks like', np.argmax(output))
    show_nn(prcimg, network)

# resize an image using a library method
def lib_downsample(image):
    resized_image = re(image, (28, 28), anti_aliasing=True)
    return resized_image

# resize a 224x224 image to 28x28
def downsample(image):
    import resize
    small = image
    while small.size > 28**2:
        small = resize.process(small)
    # fig, ax = plt.subplots(3)
    # ax[0].imshow(image/255., cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # ax[1].imshow(small/255., cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # ax[2].imshow(dataset()[2][imctr]/255., cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # plt.show()
    return small

def show_image(image):
    plt.gray()
    plt.imshow(image, interpolation='nearest')
    plt.show()

def show_nn(image, CNN):
    current_image = image[:,:,np.newaxis]/255.
    CvL1_output = CNN.layers[0].forward_prop(current_image)
    MPL1_output = CNN.layers[1].forward_prop(CvL1_output)
    # ReLu_output = CNN.layers[2].forward_prop(MPL1_output)
    # CvL2_output = CNN.layers[3].forward_prop(ReLu_output)
    # MPL2_output = CNN.layers[4].forward_prop(CvL2_output)
    prediction  = CNN.layers[2].forward_prop(MPL1_output)
    ncols = 3
    fig, ax = plt.subplots(8, ncols)
    # Image
    ax[0][ncols-1].imshow(current_image, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    # Convolutions
    for i in range(CvL1_output.shape[2]):
        ax[i][0].imshow(CvL1_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Max Pooling
    for i in range(MPL1_output.shape[2]):
        ax[i][1].imshow(MPL1_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # ReLu
    # for i in range(ReLu_output.shape[2]):
    #     ax[i][2].imshow(ReLu_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Convolutions 2
    # for i in range(CvL2_output.shape[2]):
    #     ax[i][3].imshow(CvL2_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Max Pooling 2
    # for i in range(MPL2_output.shape[2]):
    #     ax[i][4].imshow(MPL2_output[:,:,i], cmap='RdYlGn', interpolation='nearest', vmin=-1, vmax=1)
    # Softmax
    ax[1][ncols-1].imshow(prediction[:,np.newaxis], cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.show()

def build_buttons(network):
    # Buttons and their respective functions.
    buttons = [
        ['Black', lambda: changeColor([0, 0, 0])],
        ['White', lambda: changeColor([255, 255, 255])],
        ['Clear', clear],
        ['Brush Larger', lambda: changebrushSize('greater')],
        ['Brush Smaller', lambda: changebrushSize('smaller')],
        # ['Save', save],
        ['Run', lambda: run(network)],
    ]
    # Making the buttons
    for index, buttonName in enumerate(buttons):
        Button(index * (buttonWidth + 10) + 10, 10, buttonWidth,
            buttonHeight, buttonName[0], buttonName[1])

def loop():
    # Game loop.
    while True:
        screen.fill((30, 30, 30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        # Drawing the Buttons
        for object in objects:
            object.process()
        # Draw the Canvas at the center of the screen
        x, y = screen.get_size()
        screen.blit(canvas, [x/4 - canvasSize[0]/2, y/2 - canvasSize[1]/2])
        # Draw the display
        # screen.blit(display, [3*x/4 - displaySize[0]/2, y/2 - displaySize[1]/2])
        # Drawing with the mouse
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            # Calculate Position on the Canvas
            dx = mx - x/4 + canvasSize[0]/2
            dy = my - y/2 + canvasSize[1]/2
            pygame.draw.circle(
                canvas,
                drawColor,
                [dx, dy],
                brushSize
            )
            # pygame.draw.circle(
            #     display,
            #     shadeColor,
            #     [dx/10, dy/10],
            #     brushSize/5
            # )
            # pygame.draw.circle(
            #     display,
            #     drawColor,
            #     [dx/10, dy/10],
            #     brushSize/5
            # )
        # Reference Dot
        pygame.draw.circle(
            screen,
            drawColor,
            [100, 100],
            brushSize,
        )
        pygame.display.flip()
        fpsClock.tick(fps)
