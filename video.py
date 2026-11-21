
from httplib2 import Credentials
import numpy as np
import cv2
import dlib
import time
from scipy.spatial import distance as dist
from imutils import face_utils
import csv



#Firebase part
from google.cloud import firestore
from google.oauth2 import service_account
import firebase_admin

from firebase_admin import db




#####
# Add the code Ive sent here
from firebase_admin import credentials
cred = credentials.Certificate('smart-companion-d295c-firebase-adminsdk-fbsvc-21d14a7df5.json'
)
			       
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://smart-companion-d295c-default-rtdb.europe-west1.firebasedatabase.app/"
})

# Also add the file Ive given to the root of this directory
#####

ref=db.reference("/")
ref.get()

db.reference("/name").get()


ref.get()


shape=[]
def cal_yawn(shape):
	top_lip = shape[50:53]
	top_lip = np.concatenate((top_lip, shape[61:64]))

	low_lip = shape[56:59]
	low_lip = np.concatenate((low_lip, shape[65:68]))

	top_mean = np.mean(top_lip, axis=0)
	low_mean = np.mean(low_lip, axis=0)

	distance = dist.euclidean(top_mean,low_mean)
	return distance

def cal_eye(shape):
    distance=[]
    top_left_eye=shape[36:40]
    bottom_left_eye=shape[41:42]
    top_left_mean=np.mean(top_left_eye,axis=0)
    bottom_left_mean=np.mean(bottom_left_eye,axis=0)

    distance.append(dist.euclidean(top_left_mean,bottom_left_mean))


    top_right_eye=shape[43:46]
    bottom_right_eye=shape[47:48]
    top_right_mean=np.mean(top_right_eye,axis=0)
    bottom_right_mean=np.mean(bottom_right_eye,axis=0)

    distance.append(dist.euclidean(top_right_mean,bottom_right_mean))

    left_eye_aspect_ratio=0
    right_eye_aspect_ratio=0

    A=dist.euclidean(shape[37],shape[41])
    B=dist.euclidean(shape[38],shape[40])
    C=dist.euclidean(shape[36],shape[39])
    left_eye_aspect_ratio=(A+B)/(2*C)
    distance.append(left_eye_aspect_ratio)

    A=dist.euclidean(shape[43],shape[47])
    B=dist.euclidean(shape[44],shape[46])
    C=dist.euclidean(shape[42],shape[45])
    right_eye_aspect_ratio=(A+B)/(2*C)
    distance.append(right_eye_aspect_ratio)

    return distance

camera = cv2.VideoCapture(0)
face_model = dlib.get_frontal_face_detector()
landmark_model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

fields = ['Yawn Flag', 'Eye Closed Flag'] 
filename = "student_record.csv"

with open(filename, 'a') as csvfile: 
	writer = csv.DictWriter(csvfile, fieldnames = fields) 
	writer.writeheader() 

count=1

yawn_thresh = 30
ptime = 0

yawn_array=[]
drowsy_array=[]
# shape=[]

while True :
	suc,cam = camera.read()
	
	img_gray = cv2.cvtColor(cam,cv2.COLOR_BGR2GRAY)
	faces = face_model(img_gray)

	# q
	
	for face in faces:
		shapes = landmark_model(img_gray,face)
		shape = face_utils.shape_to_np(shapes)
		
		#Left eye points have been identified from shapes[36:42]
		left_eye = shape[36:42]
		#right eye points have been identified from shapes[42:47]
		right_eye=shape[42:48]
		mouth = shape[48:59]
		#cv2.drawContours(cam,[left_eye,right_eye,mouth],-1,(205, 0, 0),thickness=2)
		time.sleep(1)
		

		
	if shape is None or len(shape) == 0:
		continue
	lip_dist = cal_yawn(shape)
	print("\n---- Parameters regarding mouth ----")
	print("\n Distance between top and bottom lip : "+str(lip_dist))
	eye_dist=cal_eye(shape)
	print("\n---- Parameters regarding both eyes ----\n")
	print("Distance between top and bottom left eye lids : "+str(eye_dist[0]))
	print("Distance between top and bottom right eye lids : "+str(eye_dist[1]))
	print("EAR left eye : "+str(eye_dist[2]))
	print("EAR right eye : "+str(eye_dist[3]))

	earThreshold=(eye_dist[2]+eye_dist[3])/2

	# From data acquired from a research article 
	if(earThreshold)<0.25:
		print("Eyes closed")
		eye_closed_flag=0
		cv2.putText(cam, "Eyes Closed", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
	else:
		print("Eyes OPEN")
		eye_closed_flag=1
		cv2.putText(cam, "Eyes Open", (30, 60), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)

	print("\n")

	yawn_flag=0
	
	if lip_dist > yawn_thresh :
			print("Person yawning")
			yawn_flag=1
			cv2.putText(cam, "Yawning Detected", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 140, 255), 2)


	mydict =[{'Yawn Flag': yawn_flag, 'Eye Closed Flag': eye_closed_flag}] 
	
	yawn_array.append(yawn_flag)
	drowsy_array.append(eye_closed_flag)
	
	db.reference("/yawn").set({'yawn_array':yawn_array})
	db.reference("/drowsy").set({'drowsy_array':drowsy_array})

	
	with open(filename, 'a') as csvfile: 
		writer = csv.DictWriter(csvfile, fieldnames = fields) 
		writer.writerows(mydict) 

	count=count+1

	cv2.imshow('Webcam' , cam)
	if cv2.waitKey(1) & 0xFF == ord('q') :
		break

camera.release()
cv2.destroyAllWindows()

# Get data
yawn_array = db.reference("/yawn/yawn_array").get()
drowsy_array = db.reference("/drowsy/drowsy_array").get()

# Handle missing or empty data
if not yawn_array or not drowsy_array:
    print("No data found in Firebase.")
    exit()

total_duration = len(drowsy_array)  # in seconds

# Count blinks (eye closed is 0)
blinks = drowsy_array.count(0)
blink_rate = blinks / (total_duration / 60)  # per minute

# Count yawns (yawn flag is 1)
yawns = yawn_array.count(1)

# Print summary
print(f"Total duration: {total_duration} seconds")
print(f"Blink count: {blinks}")
print(f"Blink rate: {blink_rate:.2f} blinks/min")
print(f"Yawn count: {yawns}")

# Determine attention status
if blink_rate > 16:
    print("Student is NOT attentive (high blink rate).")
if yawns > total_duration * 0.1:
    print("Student is NOT attentive (frequent yawning).")
if blink_rate <= 16 and yawns <= total_duration * 0.1:
    print("Student appears attentive.")    








    
 

    


# %%
