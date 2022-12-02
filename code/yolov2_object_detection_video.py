import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Classes
classes = ["caneta"]
print(classes)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
cap=cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('../test/detected/test_detect_pen_video.avi', fourcc, 5, size)

while (cap.isOpened()):
    ret,frame= cap.read()
    height, width, channels = frame.shape
    #Detectando objetos
    # Funcao blob - basicamente extrai as caracteristicas dos objs
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True)
    net.setInput(blob) # Passar as caracteristicas obtidas no blob na rede
    outs = net.forward(output_layers) # Extrair todas as informacoes do objeto captado
    #print(outs[1])
    
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                #Objeto detectato
                 center_x= int(detection[0]*width)
                 center_y= int(detection[1]*height)
                 w = int(detection[2]*width)
                 h = int(detection[3]*height)
                 # Coord retang
                 x=int(center_x - w/2)
                 y=int(center_y - h/2)
                 boxes.append([x,y,w,h]) # Pegando as areas do retang
                 confidences.append(float(confidence)) # Porcentagem do obj em relacao a confianca
                 class_ids.append(class_id) #classe do obj

    # Eliminar caixas duplas do mesmo objeto
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)
    # Fazer os boundigboxes
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
    cv2.imshow("Image",frame)
    out_video.write(frame)
    if cv2.waitKey(1) == 27: #esc key stops
        break
cap.release()
cv2.destroyAllWindows()