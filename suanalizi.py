import cv2
import numpy as np

# Video akışını başlat
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü HSV renk uzayına dönüştür
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi tonlarını belirlemek için bir HSV aralığı tanımlayın
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Mavi tonlarını içeren pikselleri bulun
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Mavi piksellerin yoğunluğunu hesaplayın
    blue_pixel_count = cv2.countNonZero(blue_mask)

    # Mavi piksel sayısını görüntüye yazdırın
    cv2.putText(frame, f"Blue Pixels: {blue_pixel_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    #Eşik değerini belirleyin
    threshold = 100000
    
    if blue_pixel_count > threshold:
        cv2.putText(frame,"Temiz Su" , (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    else:
        cv2.putText(frame,"Kirli Su" , (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Videoyu göster
    cv2.imshow('Suyun Rengi Analizi', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
