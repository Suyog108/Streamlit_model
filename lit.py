import numpy as np
import cv2
import tempfile
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import imutils
from streamlit_image_select import image_select

from keras.models import load_model
from keras.preprocessing import image

from agegender_demo import BG_COLOR, IMAGE_FILES, MASK_COLOR
mp_selfie_segmentation = mp.solutions.selfie_segmentation
stframe = st.empty()

def iou(box1,box2):
    tb = min(box1[0]+0.5*box1[2],box2[0]+0.5*box2[2])-max(box1[0]-0.5*box1[2],box2[0]-0.5*box2[2])
    lr = min(box1[1]+0.5*box1[3],box2[1]+0.5*box2[3])-max(box1[1]-0.5*box1[3],box2[1]-0.5*box2[3])
    if tb < 0 or lr < 0 : intersection = 0
    else : intersection =  tb*lr
    return intersection / (box1[2]*box1[3] + box2[2]*box2[3] - intersection)

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    output_image = np.where(condition, fg_image, bg_image)
    cv2.imwrite('/tmp/selfie_segmentation_output' + str(idx) + '.png', output_image)

def interpret_output_yolov2(output, img_width, img_height):
    anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

    netout=output
    nb_class=1
    obj_threshold=0.4
    nms_threshold=0.3

    grid_h, grid_w, nb_box = netout.shape[:3]

    size = 4 + nb_class + 1;
    nb_box=5

    netout=netout.reshape(grid_h,grid_w,nb_box,size)

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = bounding_box(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    result = []
    for i in range(len(boxes)):
        if(boxes[i].classes[0]==0):
            continue
        predicted_class = "face"
        score = boxes[i].score
        result.append([predicted_class,(boxes[i].xmax+boxes[i].xmin)*img_width/2,(boxes[i].ymax+boxes[i].ymin)*img_height/2,(boxes[i].xmax-boxes[i].xmin)*img_width,(boxes[i].ymax-boxes[i].ymin)*img_height,score])

    return result

class bounding_box:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2<x3:
            return 0
        else:
            return min(x2,x4) - x3
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)

#crop
def crop(x,y,w,h,margin,img_width,img_height):
    xmin = int(x-w*margin)
    xmax = int(x+w*margin)
    ymin = int(y-h*margin)
    ymax = int(y+h*margin)
    if xmin<0:
        xmin = 0
    if ymin<0:
        ymin = 0
    if xmax>img_width:
        xmax = img_width
    if ymax>img_height:
        ymax = img_height
    return xmin,xmax,ymin,ymax

#display result
def show_image(img,results, img_width, img_height, model_age):
    img_cp = img.copy()
    for i in range(len(results)):
        #display detected face
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2

        if(w<h):
            w=h
        else:
            h=w

        xmin,xmax,ymin,ymax=crop(x,y,w,h,1.0,img_width,img_height)

        #cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(125,125,125),2)
        #cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
        cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

        target_image=img_cp

        #analyze detected face
        xmin2,xmax2,ymin2,ymax2=crop(x,y,w,h,1.1,img_width,img_height)

        face_image = img[ymin2:ymax2, xmin2:xmax2]

        if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
            continue

        #cv2.rectangle(target_image, (xmin2,ymin2), (xmax2,ymax2), color=(0,0,255), thickness=3)

        offset=16

        lines_age=open('words/agegender_age_words.txt').readlines()

        def class_labels_reassign(age):
            if 1 <= age <= 4: #(0-2)
                return 0
            elif 5 <= age <= 9:#(4-6)
                return 1
            elif 10 <= age <= 15:#(8-12)
                return 2
            elif 16 <= age <= 22:#(15-20)
                return 3
            elif 22 <= age <= 35:#(25-32)
                return 4
            elif 36 <= age <= 45:#(38-43)
                return 5
            elif 46 <= age <= 55:#(48-53)
                return 6
            else:
                return 7

        if(model_age!=None):
            shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
            img_keras = cv2.resize(face_image, (shape[1],shape[2]))
            #img_keras = img_keras[::-1, :, ::-1].copy()	#BGR to RGB
            img_keras = np.expand_dims(img_keras, axis=0)
            img_keras = img_keras / 255.0

            ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

            pred_age_keras = model_age.predict(img_keras)[0]
            prob_age_keras = np.max(pred_age_keras)
            cls_age_keras = pred_age_keras.argmax()

            age = ageList[class_labels_reassign(cls_age_keras)]

            #label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]

            cv2.putText(target_image, "Age : "+age, (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 255));
            offset=offset+16

    RGB_image = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
    st.image(RGB_image)
    #cv2.imshow('YoloKerasFaceDetection',img_cp)

#display result
def show_video(img,results, img_width, img_height, model_age):
    img_cp = img.copy()
    for i in range(len(results)):
        #display detected face
        x = int(results[i][1])
        y = int(results[i][2])
        w = int(results[i][3])//2
        h = int(results[i][4])//2

        if(w<h):
            w=h
        else:
            h=w

        xmin,xmax,ymin,ymax=crop(x,y,w,h,1.0,img_width,img_height)

        #cv2.rectangle(img_cp,(xmin,ymin),(xmax,ymax),(125,125,125),2)
        #cv2.rectangle(img_cp,(xmin,ymin-20),(xmax,ymin),(125,125,125),-1)
        cv2.putText(img_cp,results[i][0] + ' : %.2f' % results[i][5],(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

        target_image=img_cp

        #analyze detected face
        xmin2,xmax2,ymin2,ymax2=crop(x,y,w,h,1.1,img_width,img_height)

        face_image = img[ymin2:ymax2, xmin2:xmax2]

        if(face_image.shape[0]<=0 or face_image.shape[1]<=0):
            continue

        #cv2.rectangle(target_image, (xmin2,ymin2), (xmax2,ymax2), color=(0,0,255), thickness=3)

        offset=16

        lines_age=open('words/agegender_age_words.txt').readlines()

        def class_labels_reassign(age):
            if 1 <= age <= 4: #(0-2)
                return 0
            elif 5 <= age <= 9:#(4-6)
                return 1
            elif 10 <= age <= 15:#(8-12)
                return 2
            elif 16 <= age <= 22:#(15-20)
                return 3
            elif 23 <= age <= 35:#(25-32)
                return 4
            elif 36 <= age <= 45:#(38-43)
                return 5
            elif 46 <= age <= 55:#(48-53)
                return 6
            else:
                return 7

        if(model_age!=None):
            shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
            img_keras = cv2.resize(face_image, (shape[1],shape[2]))
            #img_keras = img_keras[::-1, :, ::-1].copy()	#BGR to RGB
            img_keras = np.expand_dims(img_keras, axis=0)
            img_keras = img_keras / 255.0

            ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

            pred_age_keras = model_age.predict(img_keras)[0]
            prob_age_keras = np.max(pred_age_keras)
            cls_age_keras = pred_age_keras.argmax()

            age = ageList[class_labels_reassign(cls_age_keras)]

            #label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]

            cv2.putText(target_image, "Age : "+age, (xmin2,ymax2+offset), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,0,250));
            offset=offset+16

    img_result = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
    stshowvideo = stframe.image(img_result)
    #cv2.imshow('YoloKerasFaceDetection',img_cp)

def image_input(image):
    print(image.dtype)
    bg_height, bg_width, _ = image.shape

    #frame = cv2.imread("images/dress3.jpg")
    if (image is not None):
        img = image[...,::-1]  #BGR 2 RGB
        inputs = img.copy() / 255.0
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_camera = cv2.resize(inputs, (416,416))
        img_camera = np.expand_dims(img_camera, axis=0)
        out2 = model_face.predict(img_camera)[0]
        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
        show_image(img_cv, results,  img.shape[1], img.shape[0], model_age)
    else:
        print("No frame found")

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        bg_image = None
    
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results2 = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
            (results2.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., 
        bg_image = cv2.imread('bg2.jpg')
        bg_image = cv2.resize(bg_image, (bg_width, bg_height))
        #   b) Blur the input image by applying image filtering, e.g.,
        #   bg_image = cv2.GaussianBlur(image,(55,55),0)

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)

def video_input(frame):
    print(frame.dtype)
    fr_height, fr_weidth, _ = frame.shape
    #frame = cv2.imread("images/dress3.jpg")
    if (frame is not None):
        img=frame
        img = img[...,::-1]  #BGR 2 RGB
        inputs = img.copy() / 255.0
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_camera = cv2.resize(inputs, (416,416))
        img_camera = np.expand_dims(img_camera, axis=0)
        out2 = model_face.predict(img_camera)[0]
        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
        show_video(img_cv, results, img.shape[1], img.shape[0], model_age)
    else:
        print("No frame found")

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        bg_image = None
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results2 = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
            (results2.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., 
        bg_image = cv2.imread('bg2.jpg')
        bg_image = cv2.resize(bg_image, (fr_weidth, fr_height))
        #   b) Blur the input image by applying image filtering, e.g.,
        #   bg_image = cv2.GaussianBlur(image,(55,55),0)

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
    

def image_segment(image, bg_image):
    image = cv2.resize(image, (1280,720))

    #frame = cv2.imread("images/dress3.jpg")
    if (image is not None):
        img=image
        img = img[...,::-1]  #BGR 2 RGB
        inputs = img.copy() / 255.0
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_camera = cv2.resize(inputs, (416,416))
        img_camera = np.expand_dims(img_camera, axis=0)
        out2 = model_face.predict(img_camera)[0]
        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
    else:
        print("No frame found")

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
        bg_image
    
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results2 = selfie_segmentation.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw selfie segmentation on the background image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        condition = np.stack(
            (results2.segmentation_mask,) * 3, axis=-1) > 0.1
        # The background can be customized.
        #   a) Load an image (with the same width and height of the input image) to
        #      be the background, e.g., 
        bg_image = cv2.imread(bg_image)
        bg_image = cv2.resize(bg_image, (1280, 720))
        #   b) Blur the input image by applying image filtering, e.g.,
        #   bg_image = cv2.GaussianBlur(image,(55,55),0)

        if bg_image is None:
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
        output_image = np.where(condition, image, bg_image)
        img_segment = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    #show_image(output_image, results,  img.shape[1], img.shape[0], model_age, model_gender)
    st.image(img_segment)

MODEL_ROOT_PATH="./pretrain/"
#Load Model
model_face = load_model(MODEL_ROOT_PATH+'yolov2_tiny-face.h5')
#model_age = load_model(MODEL_ROOT_PATH+'agegender_age_squeezenet.hdf5')
model_age = load_model(MODEL_ROOT_PATH+'agegender_age101_squeezenet.hdf5')

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

def main():
    # Face Analysis Application
    st.markdown(
        """<script src="https://unpkg.com/amazon-kinesis-video-streams-webrtc/dist/kvs-webrtc.min.js"></script>""",
        unsafe_allow_html=True
    )
    st.title("Age Detection Application")
    activiteis = ["Home", "Webcam Face Detection","Video Detection", "Image Detection", "Image Segmentation"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed computer vision model""")
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Build using OpenCV, Mediapipe and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has following functionalities.

                 1. Face detection Employing advanced algorithms to identify and locate faces within images or video streams for enhanced visual recognition.

                 2. Age identification Utilizing cutting-edge technology to estimate and determine the age of individuals based on facial features and characteristics.

                 3. Live webcam detection Enabling real-time analysis and detection through webcam feeds, ensuring dynamic and immediate response to visual input.

                 4. Image segmentation Employing sophisticated techniques to partition images, facilitating detailed understanding of visual content for segmenting person from the image.

                 """)
        
    elif choice == "Webcam Face Detection":
        st.header("Webcam Live Feed")
        st.write("")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is None:
                st.write("frame not found")
            else:
                #frame = cv2.imread("images/dress3.jpg")
                img=frame
                img = img[...,::-1]  #BGR 2 RGB
                inputs = img.copy() / 255.0
                img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_camera = cv2.resize(inputs, (416,416))
                img_camera = np.expand_dims(img_camera, axis=0)
                out2 = model_face.predict(img_camera)[0]
                results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                show_video(img_cv, results, img.shape[1], img.shape[0], model_age)
        #webrtc_streamer(key="example",video_transformer_factory=videoframe)

    elif choice == "Video Detection":   
        st.subheader("Upload File for Identification")
        uploaded_file = st.file_uploader("", type=["mp4","avi"])
        if uploaded_file is not None:
            st.sidebar.header("Default Video")
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            st.sidebar.video(uploaded_file)
            
            if st.sidebar.button('Detect Video Frame'):
                while cap.isOpened():                    
                    ret, frame = cap.read()
                    if ret is None:
                        st.write("frame not found")
                    else:
                        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        resize_frame = imutils.resize(frame, height=680)
                        img=resize_frame
                        img = img[...,::-1]  #BGR 2 RGB
                        inputs = img.copy() / 255.0
                        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        img_camera = cv2.resize(inputs, (416,416))
                        img_camera = np.expand_dims(img_camera, axis=0)
                        out2 = model_face.predict(img_camera)[0]
                        results = interpret_output_yolov2(out2, img.shape[1], img.shape[0])
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        show_video(img_cv, results, img.shape[1], img.shape[0], model_age)
                        #video_input(resize_frame)
                    
    elif choice == "Image Detection":
        st.subheader("Upload File for Identification")
        uploaded_file = st.file_uploader("", type=["jpg","jpeg", "png"])

        #image_file = st.file_uploader("Upload image", type=["jpg", "jpeg"])
        if uploaded_file is not None:
            st.sidebar.header("Default Image")
            st.sidebar.image(uploaded_file)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            resize_img = imutils.resize(opencv_image, height=680)
            segimg = image_input(resize_img)

    elif choice == "Image Segmentation":
        st.subheader("Upload File for Segmenation ")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg"])
        bg_select = image_select("Backgorund Images",["bg1.jpg", "bg2.jpg"])
        #bg_select = imutils.resize(bg_select, height=upimg_height)

        if uploaded_file is not None:
            st.sidebar.header("Default Image")
            st.sidebar.image(uploaded_file)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            resize_img = imutils.resize(opencv_image, height=680)
            segimg = image_segment(resize_img, bg_select)

    else:
        pass

if __name__ == "__main__":
    main()
