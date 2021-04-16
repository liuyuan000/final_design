import threading
from time import sleep
import cv2

event = threading.Event()
play = threading.Event()

def video_read(path):
    global fram, num
    num = 1
    event.clear()
    play.set()
    cap = cv2.VideoCapture(path)#打开视频
    ret,fram = cap.read()#读取视频返回视频是否结束的bool值和每一帧的图像
    while ret:
        ret,fram = cap.read()#读取视频返回视频是否结束的bool值和每一帧的图像
        num = 1 - num
        event.set()
        sleep(0.03)

    play.clear()


def show(name):
    global fram
    while play.is_set():
        if event.is_set():
            cv2.imshow(name, fram)
            cv2.waitKey(10)
        else:
            event.wait()

def save_img(path, save_path):
    global fram, num
    last_num = 0
    vid_cap = cv2.VideoCapture(path)#打开视频
    fourcc = 'mp4v'  # output video codec
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_cap.release()
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    while play.is_set():
        if event.is_set():
            if last_num ==  num:
                last_num = 1 - num
                vid_writer.write(fram)
        else:
            event.wait()

    vid_writer.release()
    

global fram


path = '/home/liuyuan/final_design/目标检测/video/MOT16-08-raw.mp4'
save_path = '/home/liuyuan/桌面/test.mp4'
video_read = threading.Thread(target=video_read,args=(path, ))
video_read.start()

show = threading.Thread(target=show,args=('result',))
show.start()

save_img = threading.Thread(target=save_img,args=(path, save_path))
save_img.start()


'''
def lighter():
    count = 0
    event.set()         #初始者为绿灯
    while True:
        if 5 < count <=10:
            event.clear()  #红灯，清除标志位
            print("\33[41;lmred light is on...\033[0m]")
        elif count > 10:
            event.set()    #绿灯，设置标志位
            count = 0
        else:
            print('\33[42;lmgreen light is on...\033[0m')

        time.sleep(1)
        count += 1


def car(name):
    while True:
        if event.is_set():     #判断是否设置了标志位
            print('[%s] running.....'%name)
            time.sleep(1)
        else:
            print('[%s] sees red light,waiting...'%name)
            event.wait()
            print('[%s] green light is on,start going...'%name)

# startTime = time.time()
light = threading.Thread(target=lighter,)
light.start()

car = threading.Thread(target=car,args=('MINT',))
car.start()
endTime = time.time()
# print('用时：',endTime-startTime)
'''