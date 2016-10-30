import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

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

def region_of_interest(img, vertices):
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
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

def draw_poly_lines(img, lines, color=[0, 0, 255], thickness=10):
    points_left_x =np.array([])
    points_left_y =np.array([])
    left_num = 0
    points_right_x =np.array([])
    points_right_y =np.array([])
    right_num = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            if ((y2-y1)/(x2-x1)<0):
                left_num = left_num + 2;
                points_left_x = np.append(points_left_x, [x1,x2])
                points_left_y = np.append(points_left_y, [y1,y2])
            if ((y2-y1)/(x2-x1)>0):	
                right_num = right_num + 2;
                points_right_x = np.append(points_right_x, [x1,x2])
                points_right_y = np.append(points_right_y, [y1,y2])
            print (str(x1)+' '+str(y1)+' '+str(x2)+' '+str(y2))
    
    print (points_left_y.shape)
    print (points_left_x.shape)
    k_left = np.poly1d(np.polyfit(points_left_y, points_left_x, 3))
    k_right = np.poly1d(np.polyfit(points_right_y, points_right_x, 3))

    points_left = np.array([])
    points_right = np.array([])
    num = 0
    for y_tem in range(350,img.shape[0]):
        num = num + 1
        x_left = k_left(y_tem)
        x_right = k_right(y_tem)
        points_left = np.append(points_left, [x_left,y_tem])
        points_right = np.append(points_right, [x_right,y_tem])
    #print (points_left.shape)
    #print (points_right.shape)
    points_left = points_left.reshape(num,2)
    points_right = points_right.reshape(num,2)
    #print (points_left)
    #sys.exit()            
    cv2.polylines(img, np.int32([points_left]),False, thickness=thickness, color=color)
    cv2.polylines(img, np.int32([points_right]),False, thickness= thickness, color=color)

def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    k_left = 0
    k_right = 0
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
            if ((y2-y1)/(x2-x1)<0):
                x1_left = x1_left + x1
                y1_left = y1_left + y1
                x2_left = x2_left + x2
                y2_left = y2_left + y2

                line_left = line_left + 1
            elif ((y2-y1)/(x2-x1)>0):
                x1_right = x1_right + x1
                y1_right = y1_right + y1
                x2_right = x2_right + x2
                y2_right = y2_right + y2
                line_right = line_right + 1
    
    x1_left = x1_left/line_left
    y1_left = y1_left/line_left
    x2_left = x2_left/line_left
    y2_left = y2_left/line_left

    x1_right = x1_right/line_right
    y1_right = y1_right/line_right
    x2_right = x2_right/line_right
    y2_right = y2_right/line_right

    k_left = (y2_left-y1_left)/(x2_left-x1_left)
    k_right = (y2_right-y1_right)/(x2_right-x1_right)

    
    y1_left_fin = 325
    x1_left_fin = x2_left - (y2_left-y1_left_fin)/k_left
    y2_left_fin = img.shape[0]
    x2_left_fin = x2_left - (y2_left-y2_left_fin)/k_left

    y1_right_fin = 325
    x1_right_fin = x2_right - (y2_right-y1_right_fin)/k_right
    y2_right_fin = img.shape[0]
    x2_right_fin = x2_right - (y2_right-y2_right_fin)/k_right    
    
    cv2.line(img, (int(x1_left_fin), int(y1_left_fin)), (int(x2_left_fin), int(y2_left_fin)), color, thickness)
    cv2.line(img, (int(x1_right_fin), int(y1_right_fin)), (int(x2_right_fin), int(y2_right_fin)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros(img.shape, dtype=np.uint8)
    line_colorimge = np.dstack((line_img,line_img,line_img))
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




os.listdir("test_images/")

for images_name in os.listdir("test_images/"):
    # Read in and grayscale the image
    image = (mpimg.imread("test_images/"+images_name)).astype('uint8')
    gray = grayscale(image)

    # Define our color criteria
    red_threshold = 100
    green_threshold = 100
    blue_threshold = 255
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = gaussian_noise(gray,kernel_size)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    
    
    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(500, 250), (500, 250), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges,vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    
    lines_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    # Create a "color" binary image to combine with line image

    color_edges = np.dstack((edges,edges,edges))
    lines_edges = weighted_img(color_edges,lines_image,0.8, 1., 0)
    cv2.imwrite("results/"+images_name,lines_edges  )   