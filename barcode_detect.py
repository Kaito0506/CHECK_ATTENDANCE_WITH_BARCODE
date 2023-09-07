import cv2
from pyzbar import pyzbar
# import ultis
import pygame
pygame.init()

pygame.mixer.init()
correct = pygame.mixer.Sound("./sound/correct.mp3")
invalid = pygame.mixer.Sound("./sound/invalid.mp3")

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
webcam = False


attendance_set = set()  # Set to store attendance information
count = 0
while True:
    if webcam:
        ret, frame = cap.read()
    else:
        frame = cv2.imread(r'images\full_cap.jpg')

    # frame = ultis.detect_barcode(frame, show=False)
    
    barcodes = pyzbar.decode(frame)
    for barcode in barcodes:
        x, y, w, h = barcode.rect
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        data = barcode.data.decode("utf-8")
        cv2.putText(frame, data, (x, y-10), font, 0.5, (0, 255, 0), 2)
        
        if data not in attendance_set:
            count+=1
            correct.play()
            attendance_set.add(data)
            with open('attendance.txt', 'a') as f:
                f.write(data + '\n')
            print("saved " + str(count) + " names")
        else:
            invalid.play()
            print("the name {} is already in the list".format(data))
    
    cv2.imshow("image", frame)
    
    if cv2.waitKey(10) & 0xFF == 27:
        break

cv2.destroyAllWindows()
