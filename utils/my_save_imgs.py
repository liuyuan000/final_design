import cv2

def save_local_person(x, img, frame_num, save_path):
    '''
    >>> x 位置

    >>> img 图片

    >>> frame_num 第几帧

    >>> save_path 保存位置
    '''
    raw = img
    left, top, right, bottom = int(x[0]), int(x[1]), int(x[2]), int(x[3])
    cv2.imwrite(save_path + 'f{}tl{}_{}br{}_{}.jpg'.format(frame_num, left, right, right, bottom), \
                                    raw[top:bottom, left:right, :])