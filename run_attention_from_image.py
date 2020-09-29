import time
import numpy as np
import re
import math
import os
import io
import pickle
from imutils.object_detection import non_max_suppression
import cv2
import pytesseract


# Load east model, checkpoints etc
try :
    if onestepdecoder:
        print('Checkpoint already loaded')
except :
    # Import attention classes
    from attention_model import *
    # Load pickel file of stored variable
    inp_lang, targ_lang, max_length_inp, max_length_targ, inp_vocab_size, \
        out_vocab_size, _, _ = pickle.load(open("all_data_att.pkl", 'rb')) 
 
    # Define local variable
    inp_embedding_dim = 100
    out_embedding_dim = 100
    inp_lstm_size = 512
    dec_unit = 512
    att_units = 512
    
    # Declare objects of required classes
    optimizer = tf.keras.optimizers.Adam()
    onestepdecoder = One_Step_Decoder(out_vocab_size, out_embedding_dim, max_length_targ, dec_unit, att_units)
    encoder = Encoder_att(inp_vocab_size, inp_embedding_dim, inp_lstm_size,max_length_inp)
    checkpoint_dir = './checkpoints_att'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_att = tf.train.Checkpoint(encoder = encoder, onestepdecoder = onestepdecoder, optimizer=optimizer)
    
    # restoring the latest checkpoint in checkpoint_dir
    checkpoint_att.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Define function to handle mutiple lines 
def maskOrig(fullTxt, k1, k2, all_recog_text, all_orig_text,all_bx, orig ):
    
    	# Translate the merged text
	textTranslated_mul = translate_sent(fullTxt, encoder, onestepdecoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
	textTranslated_mul = textTranslated_mul.strip()
	#print(textTranslated_mul)
    
    	# Get the percentages of the recognized text line size
	percent_size = []
	for ol in all_recog_text[k1:k2+1]:
		percent_size.append(len(ol)/len(fullTxt))
    
	rem_text = textTranslated_mul
	len_trans = len(textTranslated_mul)
	multi_lines = []
    	# Based on the above percentage split the translated text
	for per, pSize in enumerate(percent_size):
		cutLen = math.ceil(len_trans*pSize)
		if per != len(percent_size)-1:
			multi_lines.append(rem_text[0:cutLen])
		else:
			multi_lines.append(rem_text)
		rem_text = rem_text[cutLen: ]
    
	for sub in range(1,len(multi_lines)):
		if multi_lines[sub][0] != ' ':
			multi_lines[sub-1] = multi_lines[sub-1] + multi_lines[sub].split(" ")[0]
			multi_lines[sub] = multi_lines[sub].replace(multi_lines[sub].split(" ")[0],"",1)
      
    	# Move each the splited text to orignal image space
	for lnNo, tLine in enumerate(multi_lines):
        
		tLine = tLine.strip()
                    
        	# Get the size of text line
		(text_width, text_height) = cv2.getTextSize(tLine, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)[0]
        
        	# Create three masks of same size of the text line
		mask = np.zeros((text_height+15, text_width+50), dtype=np.uint8)
		mask1 = np.zeros((text_height+15, text_width+50), dtype=np.uint8)
		mask2 = np.zeros((text_height+15, text_width+50), dtype=np.uint8)
        
        	# Get the background color of the text
		getBgColor = GetBGColor(all_orig_text[k1:k2+1][lnNo])
		brg = getBgColor.detect()
        
        	# Move all BRG color values to all three masks 
		mask[:,:] = brg[0]
		mask1[:,:] = brg[1]
		mask2[:,:] = brg[2]
        	# Merge the mask values
		mer_mask = cv2.merge((mask, mask1, mask2))
		o_startX, o_startY, o_endX, o_endY, o_angle, o_an_h = all_bx[k1:k2+1][lnNo]
        
        	# Create blank mask of same text recognized shape
		bnk_masked = cv2.resize(mer_mask, (o_endX-o_startX+2*boundary, o_endY-o_startY+2*boundary))
		bnk_masked1 = cv2.resize(mer_mask, (o_endX-o_startX+2*boundary, o_endY-o_startY+2*boundary))

        	# Write the translated text to the mask
		line_mask = cv2.putText(mer_mask, tLine , (0, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0), 1, cv2.LINE_AA)
        
        	# Resize the mask to the original recognized text size
		if o_angle != 0:
            		# If its a angled text then first resize the masked text to the same size as whole rect around this angled text
			(h_,w_)=bnk_masked1.shape[:2]
			o_an_h = math.ceil(o_an_h/2)*2
			line_mask = cv2.resize(line_mask, (w_-10, o_an_h))
            		# Move the masked text to the rect
			bnk_masked1[(h_//2)-o_an_h//2:(h_//2)+o_an_h//2, 5:w_-5] = line_mask
            		# Rotate the rect to the opposite direction to the translated text in same angled as in recognized text
			bnk_masked1_nt = cv2.bitwise_not(bnk_masked1)
			(h_, w_) = bnk_masked1_nt.shape[:2]
			center = (w_ // 2, h_ // 2)
			M = cv2.getRotationMatrix2D(center, -o_angle, 1.0)
			rotated = cv2.warpAffine(bnk_masked1, M, (w_, h_),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            		# Replace the original text location with the new translated text
			try:
				orig[o_startY-boundary:o_endY+boundary, o_startX-boundary:o_endX+boundary] = bnk_masked						    
				orig[o_startY-boundary:o_endY+boundary, o_startX-boundary:o_endX+boundary] = rotated
            
			except:
				continue
            
		else:
            		# Replace the original text location with the new translated text
			line_mask = cv2.resize(line_mask, (o_endX-o_startX, o_endY-o_startY))

			try:

				orig[o_startY-boundary:o_endY+boundary, o_startX-boundary:o_endX+boundary] = bnk_masked		
				orig[o_startY:o_endY, o_startX:o_endX] = line_mask
			except:

				continue
	return orig


# Load tesseract and pre trained East model
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
#cap = cv2.VideoCapture('test2.mp4')
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# List all images to run the demo
file_names = []
for filename in os.listdir('text_data'):
    file_names.append('text_data/'+ filename)
tot_file = len(file_names)    
ind = 0

#while cv2.waitKey(1) < 0:
#while (cap.isOpened()):
while (cv2.waitKey(1000) != ord('q') and ind<tot_file):

    
    	# read images one by one
	#hasFrame, image = cap.read()
	image = cv2.imread(file_names[ind])
	ind += 1
    
	#if not hasFrame:
	#	break
    
    	# Get image shapes
	orig = image
	orig2 = image
	(H, W) = image.shape[:2]

    	# Declare new size for image
	(newW, newH) = (640, 320)
	rW = W / float(newW)
	rH = H / float(newH)

    	# reshape image to the new declared size
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

    	# delcrare layers we will use in our project
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]


	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

    	# Get the scores and locations in the input image
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

    	# Read through x and y values
	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			offsetX = offsetX + cos * xData1[x] + sin * xData2[x] 
			offsetY = offsetY - sin * xData1[x] + cos * xData2[x]                
        
			# calculate the UL and LR corners of the bounding rectangle
			p1x = -cos * w + offsetX
			p1y = -cos * h + offsetY
			p3x = -sin * h + offsetX
			p3y = sin * w + offsetY
                           
			# add the bounding box coordinates
			rects.append((p1x, p1y, p3x, p3y))
            
			confidences.append(scoresData[x])

    	# Get the boundaries where word may present
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	boxes = sorted(boxes, key=lambda x: (x[3],x[2]))
	bx = []
	bx_angle = []
    
    	# Go through each boxes and find the merge boxes in same line 
    	# This is to translate a whole sentence instead of single single words
	for i,(startX, startY, endX, endY) in enumerate(boxes):
        
		s_X = int(startX * rW)
		s_Y = int(startY * rH)
		e_X = int(endX * rW)
		e_Y = int(endY * rH)

		boundary = 2
        	# Get the text area
		if s_Y>10 and s_X>10:
			text = orig[s_Y-5*boundary:e_Y+5*boundary, s_X-5*boundary:e_X+5*boundary]
		else:
			text = orig[s_Y-boundary:e_Y+boundary, s_X-boundary:e_X+boundary]            
		if (text.astype(np.uint8)).size == 0:           
			continue

		text_img = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        	# Check if the text is in angled
		text_img_nt = cv2.bitwise_not(text_img)
		thresh = cv2.threshold(text_img_nt, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		coords = np.column_stack(np.where(thresh > 0))
		angle = cv2.minAreaRect(coords)[-1]
        	# Get the angle
		if angle < -45:
		    angle = -(90 + angle)
		else:
		    angle = -angle        
        
        	# If text angle is within 5 to -5 --> it is horizontal line
		if angle < 5 and angle > -5:
			ag = 0
			ah = 0
			if len(bx)==0 :
				sX = startX
				sY = startY
				eX = endX
				eY = endY
				bx.append([sX,sY,eX,eY,ag,ah])
				continue
			if endY-eY < 20:
				sX = min(startX,sX)
				sY = max(startY,sY)
				eX = max(endX,eX)
				eY = endY
				bx[-1] = [sX,sY,eX,eY,ag,ah]
			else:
				sX = startX
				sY = startY
				eX = endX
				eY = endY
				bx.append([sX,sY,eX,eY,ag,ah])
		else:
			# It is a angled text
            		# Store all angled texts, if two texts are in same angled then merge them in bx_angle
			updflag = False
			if len(bx_angle)==0:
				bx_angle.append([startX,startY,endX,endY,angle, abs(startY-endY)])
				continue
			for ibn, an in enumerate(bx_angle):
				if abs(angle-an[4]) < 12:
					sXa = min(an[0], startX)
					sYa = min(an[1], startY)
					eXa = max(an[2], endX)
					eYa = max(an[3], endY)
					bx_angle[ibn] = [sXa,sYa,eXa,eYa,angle, abs(startY-endY)]
					updflag = True
					continue
            
			if not (updflag):
				bx_angle.append([startX,startY,endX,endY,angle, abs(startY-endY)])
      
    	# Move all angled text to bx list              
	for an in bx_angle:  
		if an[4] > 0:
			bx.append([an[0]-3*boundary,an[1]-10*boundary,an[2]+3*boundary,an[3]+10*boundary,an[4],an[5]+13*boundary])
		else:
			bx.append([an[0]-3*boundary,an[1]-3*boundary,an[2]+3*boundary,an[3]+3*boundary,an[4],an[5]+3*boundary])
    
	text_recog = []
	text_pos = []
	new_pos = {}
	all_recog_text = []
	all_orig_text = []
	all_bx = []
    	# Loop through each boxes (1 box == text present in a single line)
	for k, (startX, startY, endX, endY, angle, an_h) in enumerate(bx):

		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 2
		# If angled text then crop with extra boundaries,
		if angle!=0:
			text = orig[startY-2*boundary:endY+2*boundary, startX-2*boundary:endX+2*boundary]
		else:
			text = orig[startY-boundary:endY+boundary, startX-boundary:endX+boundary]  

		if (text.astype(np.uint8)).size == 0:
			continue

		text_img = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		
		# Extract text from the box location
		config='-l eng --oem 1 --psm 3'
        
        	# If angled text then first rotate the text to its angled then use tesseract on it
        	# else directly use tesseract
		if angle != 0:
			text_img_nt = cv2.bitwise_not(text_img)
			(h_, w_) = text_img_nt.shape[:2]
			center = (w_ // 2, h_ // 2)
			M = cv2.getRotationMatrix2D(center, angle, 1.0)
			rotated = cv2.warpAffine(text_img, M, (w_, h_),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
			textRecognized = pytesseract.image_to_string(rotated, config = config, lang ='eng')           
		else:
			textRecognized = pytesseract.image_to_string(text_img, config = config, lang ='eng')
            
		textRecognized = textRecognized.strip()  
        
        	# Store each recognized text and their dimentions in list
		if (textRecognized):
			#print(textRecognized)
			all_recog_text.append(textRecognized)  
			all_orig_text.append(text)
			all_bx.append((startX, startY, endX, endY, angle, an_h))
    
	prek = 0
    	# For all recognised text, check if multi line exist and process accordinly
	for k, textRecognized in enumerate(all_recog_text):
        
        	# If its the first text recognized from image, save it
		if k == 0:
			fullTxt = textRecognized           

        	# If its the last text recognized, make last_ln true.        
		last_ln = False
		if k==len(all_recog_text)-1:
			last_ln = True
        
        	# Check if text line does'nt ends with any of ['.','?','!']
		if (textRecognized[-1] not in ['.','?','!']):
           	# if its a last text line, then send this line to 'maskOrig' function to transalate this
            		# and replace it in the original text space
			if last_ln:
				if k!= 0:
					fullTxt += ' '+ textRecognized
				orig = maskOrig(fullTxt, prek, k,all_recog_text, all_orig_text,all_bx, orig)
				fullTxt = ''
				prek = k+1
            		# If its not the last line, then check if next line starts with capital character
            		# if capital character, then it means next line is a new sentence, so send the current
            		# text line to 'maskOrig' function to transalate this and replace it in the original text space
			elif ((all_recog_text[k+1].strip())[0].isupper()):
				if k!= 0:
					fullTxt += ' '+ textRecognized
				orig = maskOrig(fullTxt, prek, k,all_recog_text, all_orig_text,all_bx, orig)
				fullTxt = ''
				prek = k+1
            
            		# if its not the last line and next line starts with small character, it means next line is 
            		# a continuation of current line, hence do no send the current line, store it for next iteration
			elif not ((all_recog_text[k+1].strip())[0].isupper()):
				if k!= 0:
					fullTxt += ' '+ textRecognized
         
        	# if text line ends with ['.','?','!'], that means current line is a end of sentence, 
        	# hence send it to 'maskOrig' function to transalate this and replace it in the original text space
		elif (textRecognized[-1] in ['.','?','!']):
			if  k!= 0:
				fullTxt += ' '+ textRecognized
			orig = maskOrig(fullTxt, prek, k,all_recog_text, all_orig_text,all_bx, orig)
			fullTxt = ''
			prek = k+1
	
    	# Show both original and transalated images
	image = cv2.resize(image, (500, 320))
	orig = cv2.resize(orig, (500, 320))
	cv2.imshow("Original text", image)
	cv2.imshow("Transalated text", orig)
	k = cv2.waitKey(30000) & 0xff
	if k == 27:    
		break

cv2.destroyAllWindows()
#cap.release()


