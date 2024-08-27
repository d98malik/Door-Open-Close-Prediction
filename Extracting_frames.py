# creating labelled data
import cv2 as c
import numpy as np

def frames_creator(video_path, OpenDoor_frame_save_path, ClosedDoor_frame_save_path):    
    cap = c.VideoCapture(video_path)
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            c.imshow("frame", frame)
            door_opening_frame = 45
            if frame_number<door_opening_frame:
                save_path = ClosedDoor_frame_save_path + "/" + f"{frame_number}.jpg"
                c.imwrite(save_path, frame)
            elif frame_number<130:
                save_path = OpenDoor_frame_save_path + "/" + f"{frame_number}.jpg"
                c.imwrite(save_path, frame)
            else:
                pass
            frame_number+=1
            if c.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    c.destroyAllWindows()
    cap.release()
    return 0


video_path = "data/Door_Opening.mp4"
OpenDoor_frame_save_path = "data/FramesExtracted/OpenDoor"
ClosedDoor_frame_save_path = "data/FramesExtracted/ClosedDoor"
frames_creator(video_path, OpenDoor_frame_save_path, ClosedDoor_frame_save_path)


