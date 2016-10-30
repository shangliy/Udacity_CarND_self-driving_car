
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math 
import sys

from moviepy.editor import VideoFileClip
from IPython.display import HTML

'''
win_left_1 = 250
win_left_2 = 550
win_left_3 = 650
win_left_4 = 400
win_right_1 = 900
win_right_2 = 650
win_right_3 = 730
win_right_4 = 1150
'''
win_left_1 = 0
win_left_2 = 0
win_left_3 = 0
win_left_4 = 0
win_right_1 = 0
win_right_2 = 0
win_right_3 = 0
win_right_4 = 0

width = 20
Height = 350
kleft_sum =0
kright_sum = 0
num_left_all = 0
num_right_all = 0
x_left_s = 0
y_left_s = 0
x_right_s = 0
y_right_s = 0
num =0




def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices_left,vertices_right):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices_left, ignore_mask_color)
    cv2.fillPoly(mask, vertices_right, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_poly_lines(img, lines, color=[0, 0, 255], thickness=10):

    global win_left_1
    global win_left_2 
    global win_left_3
    global win_left_4 
    global win_right_1 
    global win_right_2 
    global win_right_3 
    global win_right_4 
    global width
    global Height
    global kleft_sum
    global kright_sum
    global num
    global num_left_all 
    global num_right_all
    global x_left_s 
    global y_left_s 
    global x_right_s 
    global y_right_s 



    num = num + 1
    line_left = 0
    line_right = 0
    x1_left = 0
    y1_left = 0
    x2_left = 0
    y2_left = 0
    

    x1_right = 0
    y1_right = 0
    x2_right = 0
    y2_right = 0



    for line in lines:
        for x1,y1,x2,y2 in line:
            
            if ((y2-y1)/(x2-x1)<0) and (win_left_1<x1) and (x1<win_left_3) and (win_left_1<x2) and (x2<win_left_3):
                x1_left = x1_left + x1
                y1_left = y1_left + y1
                x2_left = x2_left + x2
                y2_left = y2_left + y2
                line_left = line_left + 1

            elif ((y2-y1)/(x2-x1)>0) and (win_right_2<x1) and (x1<win_right_4) and (win_right_2<x2) and (x2<win_right_4):
                x1_right = x1_right + x1
                y1_right = y1_right + y1
                x2_right = x2_right + x2
                y2_right = y2_right + y2
                line_right = line_right + 1
    if (line_left>0):
        num_left_all = num_left_all + 1       
        x1_left = x1_left/line_left
        y1_left = y1_left/line_left
        x2_left = x2_left/line_left
        y2_left = y2_left/line_left
        k_left = (y2_left-y1_left)/(x2_left-x1_left)
        kleft_sum = kleft_sum + k_left
        k_left = kleft_sum/num_left_all
        if (num_left_all < 2 or abs((x1_left+x2_left)/2 -x_left_s )<30):
            x_left_s = (x1_left+x2_left)/2
            y_left_s = (y1_left+y2_left)/2
    else:
        k_left = kleft_sum/num_left_all

    if (line_right>0):
        num_right_all = num_right_all + 1       
        x1_right = x1_right/line_right
        y1_right = y1_right/line_right
        x2_right = x2_right/line_right
        y2_right = y2_right/line_right
        k_right = (y2_right-y1_right)/(x2_right-x1_right)
        kright_sum = kright_sum + k_right
        k_right = kright_sum/num_right_all
        if (num_right_all < 2 or abs((x1_right+x2_right)/2 -x_right_s )<30):
            x_right_s = (x1_right+x2_right)/2
            y_right_s = (y1_right+y2_right)/2
    else:
        k_right = kright_sum/num_right_all


        
    #print (points_left_y.shape)
    #print (points_left_x.shape)
    
    


    points_left = np.array([])
    points_right = np.array([])
    num = 0
    for y_tem in range(Height,img.shape[0]):
        num = num + 1
        x_left = x_left_s - (y_left_s-y_tem)/k_left
        x_right = x_right_s - (y_right_s-y_tem)/k_right
        points_left = np.append(points_left, [x_left,y_tem])
        points_right = np.append(points_right, [x_right,y_tem])
    #print (points_left.shape)
    #print (points_right.shape)
    points_left = points_left.reshape(num,2)
    points_right = points_right.reshape(num,2)
    #print (points_left)
    #sys.exit() 

    
    
    cv2.polylines(img, np.int32([[(win_left_1,img.shape[0]*0.9),(win_left_2, Height), (win_left_3, Height), (win_left_4,img.shape[0]*0.9)]]),True, thickness=thickness, color=color)           
    cv2.polylines(img, np.int32([[(win_right_1,img.shape[0]*0.9),(win_right_2, Height), (win_right_3, Height), (win_right_4,img.shape[0]*0.9)]]),True, thickness=thickness, color=color)           
    
    win_left_1 =  x_left_s - (y_left_s-(img.shape[0]*0.9))/k_left-width
    win_left_2 =  x_left_s - (y_left_s-(Height))/k_left-width
    win_left_3 =  x_left_s - (y_left_s-(Height))/k_left+width
    win_left_4 =  x_left_s - (y_left_s-(img.shape[0]*0.9))/k_left+width
    win_right_1 =  x_right_s - (y_right_s-(img.shape[0]*0.9))/k_right-width
    win_right_2 =  x_right_s - (y_right_s-(Height))/k_right-width
    win_right_3 =  x_right_s - (y_right_s-(Height))/k_right+width
    win_right_4 =  x_right_s - (y_right_s-(img.shape[0]*0.9))/k_right+width
    
    cv2.polylines(img, np.int32([points_left]),False, thickness=thickness, color=color)
    cv2.polylines(img, np.int32([points_right]),False, thickness= thickness, color=color)
                          
   
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    line_colorimge = np.dstack((line_img,line_img,line_img))
    #draw_lines(line_colorimge, lines)
    draw_poly_lines(line_colorimge, lines)
    return line_colorimge

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    gray = grayscale(image)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_noise(gray,kernel_size)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 180
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    
    global win_left_1
    global win_left_2 
    global win_left_3
    global win_left_4 
    global win_right_1 
    global win_right_2 
    global win_right_3 
    global win_right_4
    global num

    if num == 0:
        
    
        win_left_1 = imshape[1]/2-400
        win_left_2 = imshape[1]/2-100
        win_left_3 = imshape[1]/2-20
        win_left_4 = imshape[1]/2-200
        win_right_1 = imshape[1]/2+200
        win_right_2 = imshape[1]/2+20
        win_right_3 = imshape[1]/2+100
        win_right_4 = imshape[1]/2+500
    
   
    vertices_left = np.array([[(win_left_1,imshape[0]*0.9),(win_left_2, Height), (win_left_3, Height), (win_left_4,imshape[0]*0.9)]], dtype=np.int32)
    vertices_right = np.array([[(win_right_1,imshape[0]*0.9),(win_right_2, Height), (win_right_3, Height), (win_right_4,imshape[0]*0.9)]], dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices_left,vertices_right)

    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments
    
    lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #return lines_image
    # Create a "color" binary image to combine with line image

    color_edges = np.dstack((edges,edges,edges))
    result = weighted_img(color_edges,lines_image,0.8, 1., 0)
    
    return result

'''
challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
'''
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
