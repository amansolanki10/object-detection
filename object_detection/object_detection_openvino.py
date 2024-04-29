from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2



inputconfidence = 0.2
movidius = 0

#list of class labels MobileNet SSD was trained 
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

ARCH_FPATH="MobileNetSSD_deploy.xml"
MODEL_FPATH="MobileNetSSD_deploy.bin"

net = cv2.dnn.readNet(ARCH_FPATH, MODEL_FPATH)
#net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD) 

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

print("Starting live video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

	net.setInput(blob)
	detections = net.forward()

	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > inputconfidence:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Video Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	fps.update()

fps.stop()
print("Time elapsed: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

