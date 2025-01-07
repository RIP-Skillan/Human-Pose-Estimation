#User-friendly library for building interactive web app
import streamlit as st 

#Other Imports
from PIL import Image       #for manipulating and working with images
import cv2                  #for image procesing
import numpy as np

#Creating an indexing for Body Parts which for feeding to the neural network
BODY_PARTS = {"Nose":0, "Neck":1, "RShoulder":2, "RElbow":3, "RWrist":4, "LShoulder":5, "LElbow":6,
              "LWrist":7, "RHip":8, "RKnee":9, "RAnkle":10, "LHip":11, "LKnee":12, 
              "LAnkle":13, "REye":14, "LEye":15, "REar":16, "LEar":17, "Background":18}

#Body Part Adjacency Pairs for connecting the frame between adjacent body parts
POSEPAIRS= [["Neck","RShoulder"], ["Neck","LShoulder"], ["RShoulder","RElbow"], ["RElbow","RWrist"],
            ["LShoulder","LElbow"], ["LElbow","LWrist"], ["Neck","RHip"], ["RHip","RKnee"], ["RKnee","RAnkle"],
            ["Neck","LHip"], ["LHip","LKnee"], ["LKnee","LAnkle"], ["Neck","Nose"], 
            ["Nose","REye"], ["REye","REar"], ["Nose","LEye"], ["LEye","LEar"]]

#Setting height and width of our image
width = 368
height = 368
inwidth = width
inheight = height

#Loading the pre-trained model into net variable
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

#Streamlit web-app design
st.title("Human Pose Estimation OpenCV")
st.text("Make sure to have a clear image with all the parts visible")

#file uploader for user to upload an image from local storage to be processed
img_file=st.file_uploader("Upload an image, Make sure you have a clear image", type = ["jpg", "jpeg", "png"])

if img_file is not None:
    image = np.array(Image.open(img_file))
else:
    image = np.array(Image.open("1.png"))           #default image file in case user doesn't upload an image

st.subheader('Original Image')
st.image(image, caption=f"Original Image")

#slider for setting the threshold for detecting the key points
thres=st.slider('Threshold for detecting the key points', min_value = 0, max_value = 100, step = 5)
thres = thres/100

#ML implementation for human-pose estimation
def poseDectector(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  #Converts input image from RGB to BGR, since the model processes in BGR
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    #Creates a 4-dimensional blob from image after resizing, rescaling and mean subtraction
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inwidth, inheight), (127.5, 127.5, 127.5), swapRB = True, crop = False))
    
    #Performs forward pass through the neural network
    out = net.forward()
    out = out[:, :19, :, :]         #Selects first 19 body parts from the output as defined in BODY_PARTS  

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]                               #HeatMap of corresponding body part
        _, conf, _, point = cv2.minMaxLoc(heatMap)              #finding max value in heatmap
        x = (frameWidth*point[0]/out.shape[3])                  #Calculating x and y coordinates of the keypoints
        y = (frameHeight*point[1]/out.shape[2])                 # in the orignial image
        points.append((int(x),int(y)) if conf>thres else None)

    for pair in POSEPAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        #Checking for pose-pairs existance in points 
        if points[idFrom] and points[idTo]:                                                 
            cv2.line(frame, points[idFrom], points[idTo], (0,255,0), 3)                     #draws a line between the body parts
            cv2.ellipse(frame, points[idFrom], (3,3), 0, 0, 360, (0, 0, 255), cv2.FILLED)   #draws a red spot to mark the body parts
            cv2.ellipse(frame, points[idTo], (3,3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    t, _ =net.getPerfProfile()

    return frame

output = poseDectector(image)

st.subheader('Positions Estimated')
st.image(output, caption = f"Posiions Estimated") #Displaying output
