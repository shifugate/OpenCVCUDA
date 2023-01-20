import cv2
import imutils
import time
import threading

running = True

def grab_video(video, type):
    vod = cv2.VideoCapture(video)

    scale = 0.5

    gpu_frame = cv2.cuda_GpuMat()

    old = None
    thresh = None
    start = None
    end = None
    window = "left"
    detection_time = time.time()
    start_time = time.time()
    end_time = time.time()

    if type == 1:
        window = "right"

    while running:
        ret, frame = vod.read()

        if frame is None:
            break

        gpu_frame.upload(frame)

        width = int(1280 * scale)
        height = int(720 * scale)

        resized = cv2.cuda.resize(gpu_frame, (width, height))

        if type == 0:
            cropped = cv2.cuda_GpuMat(resized, (0, 0, int(width / 3), height))
        else:
            cropped = cv2.cuda_GpuMat(resized, (int(width / 3 * 2), 0, int(width / 3), height))

        gray = cv2.cuda.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blur = cv2.cuda.bilateralFilter(gray,90,90,90)

        if old is not None:
            delta = cv2.cuda.absdiff(old, blur)
            thresh = cv2.cuda.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        old = blur

        if thresh is not None:
            thresh = thresh.download()
            cropped= cropped.download()

            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            detected = False

            for c in cnts:
                if cv2.contourArea(c) > 200 or cv2.contourArea(c) < 50:
                    continue

                detection_time = time.time()

                (x, y, w, h) = cv2.boundingRect(c)

                cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                if type == 0:
                    if start is None:
                        start_time = time.time()
                        start = (x - w / 2)
                    elif (x - w / 2) > start and (end is None or (x - w / 2) > end):
                        end_time = time.time()
                        end = (x - w / 2)
                else:
                    if start is None:
                        start_time = time.time()
                        start = (x - w / 2)
                    elif (x - w / 2) < start and (end is None or (x - w / 2)  < end):
                        end_time = time.time()
                        end = (x - w / 2)

                detected = True

            pass_time = time.time() - detection_time
            interval_time = end_time - start_time 

            if detected is False and end is not None and pass_time > 1:
                distance = abs(end - start)
                velocity = int ((distance / interval_time) / 20)

                print("%s | x start :%s | x end :%s | distance :%s | time :%s | velocity :%s" % (window, start, end, distance, interval_time, velocity))

                start = None
                end = None
                start_time = time.time()
                end_time = time.time()
            elif pass_time > 1:
                start = None
                end = None
                start_time = time.time()
                end_time = time.time()

            cv2.imshow(window, cropped)
            
            cv2.waitKey(1)

    vod.release()

    cv2.destroyWindow(window)

def main():
    try:
        global running

        th0 = threading.Thread(target=grab_video, args=['test.mkv', 0])
        th0.start()

        th1 = threading.Thread(target=grab_video, args=['test.mkv', 1])
        th1.start()

        input('Press Enter to close\n')
    finally:
        running = False

    running = False

if __name__ == "__main__":
    main()