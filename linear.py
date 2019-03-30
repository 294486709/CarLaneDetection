import cv2
import numpy as np


HOR_PERSENTAGE_LEFT = 0.2
HOR_PERSENTAGE_RIGHT = 0.9
VER_DOWN_PERSENTAGE = 0.90
VER_UP_PERSENTAGE = 0.6


def line_edit(Lines):
	Slope, Intercetp = Lines
	y1 = VideoFrameHeight
	y2 = int(VideoFrameHeight * 0.65)
	x1 = int((y1 - Intercetp)/Slope)
	x2 = int((y2 - Intercetp)/Slope)
	return np.array([x1, y1, x2, y2])


def average_slope_of_lines(Lines,left_prev, right_prev):
	Lhs = []
	Rhs = []
	if Lines is not None:
		for Line in Lines:
			x1, y1, x2, y2 = Line.reshape(4)
			try:
				Parameters = np.polyfit((x1, x2), (y1, y2), 1)
			except np.RankWarning:
				continue

			Slope = Parameters[0]
			if abs(Slope) < 0.1:
				continue
			Intercept = Parameters[1]
			if Slope < 0:
				Lhs.append((Slope, Intercept))
			else:
				Rhs.append((Slope, Intercept))
	else:
		return [left_prev, right_prev]
	Lhs_average = np.average(Lhs, axis=0)
	Rhs_average = np.average(Rhs, axis=0)
	try:
		left_line = line_edit(Lhs_average)
	except TypeError:
		left_line = left_prev
	try:
		right_line = line_edit(Rhs_average)
	except TypeError:
		right_line = right_prev
	return [left_line, right_line]


def frame_process(Frame, left_prev, right_prev):
	# undisort frame
	# Frame = cv2.undistort(Frame, MTX, DIST, None, MTX)
	# convert frame to gray scale
	FrameGray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
	FrameHLS = cv2.cvtColor(Frame, cv2.COLOR_BGR2HLS)
	# apply gussianblur
	FrameCanny = cv2.Canny(FrameHLS, CANNY_DOWN_TH, CANNY_UP_TH)
	# area of interest
	FrameMask = np.zeros_like(FrameGray)
	TargetPoly = np.array([(HOR_PERSENTAGE_LEFT * VideoFrameWidth, VER_DOWN_PERSENTAGE * VideoFrameHeight),
						   (HOR_PERSENTAGE_RIGHT * VideoFrameWidth, VER_DOWN_PERSENTAGE * VideoFrameHeight),
						   (0.5 * VideoFrameWidth, VER_UP_PERSENTAGE * VideoFrameHeight)], 'int32')
	cv2.fillPoly(FrameMask, [TargetPoly], 255)
	FrameMasked = cv2.bitwise_and(FrameCanny, FrameMask)
	FrameMasked1 = cv2.bitwise_and(FrameGray, FrameMask)
	# hough transformation
	Lines = cv2.HoughLinesP(FrameMasked, 1, np.pi / 180, 20, None, minLineLength=20)
	# average line
	Average_Lines = average_slope_of_lines(Lines, left_prev, right_prev)
	left_prev = Average_Lines[0]
	right_prev = Average_Lines[1]
	TempPoly = []
	Tik = False
	for Line_temp in Average_Lines:
		if len(Line_temp):
			x1, y1, x2, y2 = Line_temp.reshape(4)
			cv2.line(Frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
			if Tik:
				TempPoly.append((x2, y2))
				TempPoly.append((x1, y1))
			else:
				TempPoly.append((x1, y1))
				TempPoly.append((x2, y2))
			Tik = ~Tik
	OverLay = np.zeros_like(Frame)
	TempPoly = np.array(TempPoly, np.int32)
	cv2.fillConvexPoly(OverLay, TempPoly, (0, 255, 0))
	Frame = cv2.addWeighted(Frame, 1, OverLay, 0.3, 0)
	return Frame, left_prev, right_prev


def main():
	# global variable
	global CANNY_UP_TH
	global CANNY_DOWN_TH
	global MTX
	global DIST
	global VideoFrameWidth
	global VideoFrameHeight
	left_prev = 0
	right_prev = 0

	CANNY_UP_TH = 200
	CANNY_DOWN_TH = 50

	# read media file
	VideoFeed = cv2.VideoCapture('subject.mp4')
	VideoFlag, VideoFrame = VideoFeed.read()
	# media file check
	if not VideoFlag:
		print('Error Opening Media File!')
		exit(1)
	# new media writer
	VideoFrameRate = VideoFeed.get(cv2.CAP_PROP_FPS)
	VideoFrameWidth = int(VideoFeed.get(cv2.CAP_PROP_FRAME_WIDTH))
	VideoFrameHeight = int(VideoFeed.get(cv2.CAP_PROP_FRAME_HEIGHT))
	NewMediaEncoder = cv2.VideoWriter_fourcc(*'DIVX')
	NewMedia = cv2.VideoWriter('processed.avi', NewMediaEncoder, VideoFrameRate, (VideoFrameWidth, VideoFrameHeight))
	# process loop
	while VideoFlag:
		# frame process
		ProcesedFrame, left_prev, right_prev = frame_process(VideoFrame, left_prev, right_prev)
		# write frame to new media file
		NewMedia.write(ProcesedFrame)
		# next frame
		VideoFlag, VideoFrame = VideoFeed.read()
	NewMedia.release()
	print('Process Completed!')
	pass


if __name__ == '__main__':
	main()
