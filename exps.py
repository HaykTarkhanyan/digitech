import cv2
import matplotlib.pyplot as plt
from time import time

file_name = '2kameraeov_himnakan.MOV'
# file_name = 'trim1.mov'

# file_name = '2kameraeov_himnakan_cropped.mp4'
# cap = cv2.VideoCapture('IMG_3654.MOV')

cap = cv2.VideoCapture(file_name)
# cap = cv2.VideoCapture('trim1.mov')
# BOARD = 

def cordinate_to_board(x, y, cell_size=100):
    """
    Function takes as input x, y cordinates of the hit and
    returns cordinate of the cell where hit happened

    orinakik` cordinate_to_board(75, 120) -> (0, 2)

    """
    return (x//cell_size, y//cell_size)

def get_pixel_cords_for_position(x, y, cell_size):
    return ([cell_size*y, cell_size*(y+1)],  \
            [cell_size*x, cell_size*(x+1)])

def chignore_hit(hit_time, HIT_HISTORY, allowed_time_window):
    last_hit = HIT_HISTORY[-1]
    return (hit_time - last_hit) > allowed_time_window



print(f'file is {file_name}')

BOARD_zbaxvac = [[0,0,0], \
                 [0,0,0], \
                 [0,0,0], ]

HIT_HISTORY = [-509]
TIME_WINDOW = 2.5
SPEED_VIDEO = 1 # inchqan poqr etqan arag

CELL_SIZE = 220

print(BOARD_zbaxvac)
while cap.isOpened():
    # time.sleep(0.1)
    ret, frame =  cap.read()
    if ret == True:
        scale = 6
        frame = cv2.resize(frame, (384*scale,216*scale))
        color = frame.copy()



        if file_name.startswith('2kameraeov_himnakan'):
            frame = frame[30*scale:140*scale,118*scale:230*scale]     
            color = color[30*scale:140*scale,118*scale:230*scale]            
       
        else:
            frame = frame[40*scale:154*scale,110*scale:224*scale]
            color = color[40*scale:154*scale,110*scale:224*scale]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # frame = cv2.Canny(frame, 50, 100)
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # circle = cv2.HoughCircles(cv2.GaussianBlur(frame, (5, 5), 0), cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=100, param2=30, minRadius=0, maxRadius=0)
        for i in range(3):
            for j in range(3):
                if BOARD_zbaxvac[i][j]:
                    x_s, y_s = get_pixel_cords_for_position(i, j, cell_size=CELL_SIZE)
                    frame[x_s[0]:x_s[1], y_s[0]:y_s[1]] = 0
                    color[x_s[0]:x_s[1], y_s[0]:y_s[1]] = 0


        circle = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, dp=1, minDist=300, param1=50, param2=35, minRadius=10, maxRadius=40)
        # print(circle)
        if circle is not None:
            if chignore_hit(time(), HIT_HISTORY, TIME_WINDOW):
                for i in circle[0, :]:
                    # draw the outer circle
                    print(i)
                    x = int(i[0])
                    y = int(i[1])
                    r = int(i[2])

                    x_pos, y_pos = cordinate_to_board(x,y, cell_size=CELL_SIZE)
                    print('xpos and y_pos are')
                    print(x_pos, y_pos) 
                    print('////////////')
                    x_s, y_s = get_pixel_cords_for_position(x_pos, y_pos, cell_size=CELL_SIZE)
                    print(x_s, y_s)
                    print('x_pos is', x_pos)
                    BOARD_zbaxvac[x_pos][y_pos] = 1
                    HIT_HISTORY.append(time())

                    print(HIT_HISTORY)
                    # cv2.circle(frame, (x, y), r, 255, 2)
                    # frame[x-r:x+r, y-r:y+r] = 0

                    # frame[x_s[0]:x_s[1], y_s[0]:y_s[1]] = 0


                    # draw the center of the circle
                    cv2.circle(frame,(int(i[0]),int(i[1])),int(2),(255,0,255),3)
                    cv2.circle(color,(int(i[0]),int(i[1])),int(2),(255,0,255),3)



            edge = cv2.putText(frame, 'detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
            edge = cv2.putText(color, 'detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
      
        cv2.imshow("Frame",frame)
        cv2.imshow("Frame",color)



        if cv2.waitKey(SPEED_VIDEO) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()





### experiments starting from here

# img = frame.copy()

# contours = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     # box_list = box_list.append([[x, y, w, h]])
#     cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2)

# plt.imshow(img, cmap="gray")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(img, (5, 5), 0)
# edge = cv2.Canny(blurred, 50, 100)
# edge=cv2.putText(edge, 'detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
# plt.imshow(frame)
# x,y = edge.shape
# edge2=edge[int(x*0.3):int(x*0.7),int(y*0.3):int(y*0.7)]
# plt.imshow(cv2.GaussianBlur(edge, (5, 5), 0))

# edge.shape
# (edge2>0).sum()/edge2.size


# height, width = frame.shape[:2]
# maxRadius = int(1.1*(width/12)/2)
# minRadius = int(0.9*(width/12)/2)


# cv2.HoughCircles(cv2.GaussianBlur(frame, (5, 5), 0),cv2.HOUGH_GRADIENT, dp=1,minDist=10, param1=100,param2=30,minRadius=0,maxRadius=0)
# cv2.HoughCircles(cv2.GaussianBlur(edge, (5, 5), 0),cv2.HOUGH_GRADIENT, 1,50, param1=30,param2=50,minRadius=0,maxRadius=0)

# f=frame.copy()
# for i in circle[0,:]:
#     # draw the outer circle
#     cv2.circle(f,(i[0],i[1]),int(i[2]),(255,255,255),2)
#     # draw the center of the circle
#     # cv2.circle(frame,(i[0],i[1]),int(2),(255,255,255),3)

# plt.imshow(f, cmap="gray")
