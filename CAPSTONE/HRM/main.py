import cv2
from helpers import buildGauss, reconstructFrame, detect_face
import numpy as np


# Parameters
videoWidth = 40
videoHeight = 30
videoChannels = 3
videoFrameRate = 15

# Color Magnification Parameters
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150

# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (20, 30)
bpmTextLocation = (videoWidth//2 + 5, 30)
fontScale = 1
fontColor = (0,255,255)
lineType = 2
boxWeight = 3
# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 15
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))


def main():

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    
    # Cascade loading
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load('./haarcascade_frontalface_alt.xml'):
        print('--(!)Error loading face cascade')
        exit(0)
    bufferIndex = 0
    i = 0
    bpmBufferIndex = 0
    while (True):
        ret, frame = webcam.read()
        if ret == False:
            break

        # detect face frame
        face_frame = detect_face(frame, face_cascade)
        center = int(face_frame.shape[0]/2)

        # slice a rect out of estimated forehead
        detectionFrame = face_frame[center+40:center+70, center-20:center+20, :]

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        bufferIndex = (bufferIndex + 1) % bufferSize

        if i > bpmBufferSize:
            cv2.putText(frame, "BPM: {}".format(bpmBuffer.mean()), bpmTextLocation, font, fontScale, fontColor, lineType)
        else:
            cv2.putText(frame, "Calculating BPM...", loadingTextLocation, font, fontScale, fontColor, lineType)

        cv2.imshow("Webcam Heart Rate Monitor", frame)
        '''
        fps = webcam.get(cv2.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(CV_CAP_PROP_FPS): {0}".format(fps))
		'''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()