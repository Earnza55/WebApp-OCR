#***************************************************************
# Document Understanding using YOLO8
#***************************************************************
# https://docs.ultralytics.com/modes/predict/#working-with-results
# https://encord.com/blog/yolo-object-detection-guide/          # YOLO8 Explanation

# conda create -n yolov8_ocr python=3.10 tk
# conda activate yolov8_ocr
# conda install -c conda-forge -c pytorch -c nvidia ultralytics pytorch torchvision pytorch-cuda=11.8 
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install -c conda-forge charset-normalizer
# conda install opencv
# python yolov8_char.py predict TrainResult/documentOcr/weights/best.pt dataset_char/test manual save
#                                                ^ เปลี่ยนโมเดลตามที่ใช้     

from ultralytics import YOLO
import os, sys, torch
import gpu_status
import cv2, tkinter


# For thai label
from PIL import Image, ImageFont, ImageDraw
import re

# SAHI
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'   # Library Collision issue

cBlue=(255,0,0); cRed=(0,0,255); cOrange=(0,128,255); cDarkGray=(80,80,80); cMagenta=(255,0,255); cCyan=(255,255,0)
cGreen=(0,255,0); cYellow=(0,255,255); cLightGray=(160,160,160)
'''
clsType = ['Word', 'TextLine', 'TextBlock', 'Column', 'NumExpression', 'Symbol', 
           'Table', 'Picture', 'HorLine', 'VertLine',  'BoxLine', 'Column', 'Page' ]        
'''
clsColor = [cBlue, cRed, cOrange, cDarkGray, cMagenta, cCyan, 
            cGreen, cYellow, cLightGray, cLightGray,  cLightGray, cLightGray, cDarkGray]

clsType = ['ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 
  'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 
  'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ฤ', 'ล', 'ฦ', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 
  'ฯ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', 'ฺ',  '฿', 'เ', 'แ', 'โ', 'ใ', 'ไ', 'ๅ', 'ๆ',
   '็', '่', '้', '๊', '๋', '์', 'ํ',
  '๐', '๑', '๒', '๓', '๔', '๕', '๖', '๗', '๘', '๙',
  '!', '"', '#', '$', '%', '&', 'pL', '(', ')', '*', '+', ',', '-', '.', '/', 
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
  ':', ';', '<', '=', '>', '?', '@',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  '[', 'Won', ']', '^', '_', 'pR', 
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
  'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
  '{', 'vF', '}', '~',]

#=========================================================
# Get Screen Info.
#=========================================================
def get_screen_resolution():
    global screenWidth, screenHeight

    app = tkinter.Tk()
    screenWidth = app.winfo_screenwidth()
    screenHeight = app.winfo_screenheight()

    screenWidth = int(round((screenWidth * 2.0) /3.0,0))
    #print("width=%d, height=%d" %(width,height))
    app.destroy()

def show_program_usage():
    print("*"*80)
    print("USAGE[train]:   python yolov8.py  train    dataYmal[.yaml]  configYaml[.yaml]    epochNo ")
    print("USAGE[predict]: python yolov8.py  predict  bestWeight       DataFolder/fileName  manual/auto  save/nosave")
    print("*"*80)
      
#=========================================================
# SHow OpenCV-style Image Window
#=========================================================
def show_image(image,windowTitle="",x=0,y=0,showMode="show") :
    global screenWidth, screenHeight

    if windowTitle == "" :
        windowTitle = "Result Image"
        
    resizedImage = cv2.resize(image, (screenWidth, screenHeight), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow(windowTitle)
    cv2.moveWindow(windowTitle,x,y)
    cv2.resizeWindow(windowTitle, screenWidth, screenHeight)
    
    cv2.imshow(windowTitle,resizedImage)
        
    if showMode == 'manual':
        miliSecond = 0
    else:
        miliSecond = 10

    inKey = cv2.waitKey(miliSecond)
    cv2.destroyAllWindows()
    
    return inKey

#=========================================================
# Get DataSet File(s)/Folder Info.
#=========================================================
def train_custom_data(dataYaml="yolov8_ocr_data.yaml", cfgYaml="yolov8_ocr_config.yaml",  epochNo = 40) :

    # For Training
    model = YOLO("yolov8m.yaml")    # build a new model from scratch
    model = YOLO("yolov8m.pt")     # load a pretrained model (recommended for training)
    
    model.train(data=dataYaml, cfg=cfgYaml, imgsz=640, batch=8,  epochs=epochNo, verbose=False, device=0)  # train the model
            #data="D:/myprojects/yolov8_ocr/ultralytics/cfg/datasets/yolov8_ocr.yaml"
            #cfg="D:/myprojects/yolov8_ocr/ultralytics/cfg/yolov8_ocr.yaml"
    gpu_status.displayGpuStatus(1)

    metrics = model.val()  # evaluate model performance on the validation set
    #path = model.export(format="onnx")  # export the model to ONNX format
    
    print("*"*80)
    print("Training Ended !!!")
    print("*"*80)
    
#=========================================================
# Get DataSet File adn Predict
#=========================================================
# def predict_data(bestWeight = "best.pt", dataFolder = "dataset_ocr/test/images/", 
def predict_data(bestWeight = "best.pt", dataFolder = "dataset_char/test", 
                 showMode = 'manual', saveMode = 'save' ) :  
    cwd = os.getcwd()
    argFullPath =  os.path.join(cwd, dataFolder)
    imgDataList = []
    
    print("data_path: ",argFullPath)
    if os.path.isfile(argFullPath) :
        imgDataList.append(argFullPath)
    elif os.path.isdir(argFullPath) :
        fileList = os.listdir(argFullPath)
        imgDataList = [os.path.join(argFullPath,file) for file in fileList 
                        if (file.endswith(".jpg") or file.endswith(".png") or 
                            file.endswith(".JPG") or file.endswith(".PNG"))]

    # SAHI
    yolov8_model_path = bestWeight

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.85,
        device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    gpu_status.displayGpuStatus(1)
    '''
    if device != "cpu" :
        print("!"*80)
        model = torch.nn.DataParallel(model)
        model.to(device)
    '''

    print("* Best Weight [%s] is Used. "  %bestWeight)
    
    originalExt = imgDataList[0][-4:].upper()
    
    if originalExt == '.JPG' :
        newExt = '.png'
    elif originalExt == '.PNG' :
        newExt = '.jpg'
    else :
        print("\n\n Image Format Mismatch !!! [.jpg or .png needed]")
        exit[1]
        
    imgNum = 1
    for imgFileName in imgDataList :
        start = time.time()
        print("\nin process image no." + str(imgNum) + "  . . . .")
        # Yolo SAHI
        results = get_sliced_prediction(
            imgFileName,
            detection_model,
            slice_height = 250,
            slice_width = 250,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2
        )

        results = results.to_coco_annotations()

        # NEW CODE FOR THAI CHARACTER
        image = Image.open(imgFileName)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("Tahoma.ttf", 12)

        # write to txt file
        file_name = re.sub(".jpg", "", imgFileName)
        file_name = re.sub(".png", "", file_name)
        txt_file = open( re.sub(".jpg", "", file_name)+".txt", "w",encoding="utf-8")

        char_lines = LinkedList()

        for result in results:
            if (result["category_name"] == "word" or result["category_name"] == "special_symbol"):
                continue
            else:

                # set values
                left = result["bbox"][0]
                top = result["bbox"][1]
                right = left + result["bbox"][2]
                bottom = top + result["bbox"][3]
                confValue = result["score"]
                if (result["category_name"] == "picture"):
                    clsValue = "picture"
                else:
                    # clsValue = chr(int(result["category_name"], 16)) # result in Hexadecimal [0X.....]
                    clsValue = result["category_name"]
                # print("clsValue = " + clsValue,
                #       ", l = " + str(left),
                #       ", top = " + str(top),
                #       ", r = " + str(right),
                #       ", bottom = " + str(bottom),
                #       )
                
                # NEW CODE FOR THAI CHARACTER
                draw.rectangle((left,top,right,bottom), outline=(0,255,0), width=1)
                draw.text((left+5,top-15), clsValue, fill="red", font = font, align ="left")
                
                data = Char()
                data.character = clsValue
                data.t = top
                data.l = left
                data.b = bottom
                data.r = right
                data.x_center,data.y_center = getCenterPoint(left, right ,top, bottom)

                char_lines.insert(data)

    
        if saveMode == 'save':
            resultFileName = imgFileName.replace(originalExt,newExt,1)

            # NEW CODE FOR THAI CHARACTER
            image.save(resultFileName)

            # CODE FOR TXT OUTPUT FILE

            # print("\nResult: ")
            # char_lines.printLL()
            # print("After sorted: ")

            sorted_char_lines = char_lines.sort()
            for node in sorted_char_lines:
                if (node is None):
                    continue
                # print(node.childList.getTextLine(), end="")
                txt_file.write(node.childList.getTextLine())
            

        else :
            resultFileName = ""
            sorted_char_lines = char_lines.sort()
            for node in sorted_char_lines:
                if (node is None):
                    continue
                # print(node.childList.getTextLine(), end="")
                txt_file.write(node.childList.getTextLine())
        end = time.time()
        prediction_time = "{:.2f}".format(round((end - start), 2))
        print("Image no."+ str(imgNum) + " prediction time: " + prediction_time + " s")
        imgNum += 1

    print("\n","*"*80)
    print("Prediction Ended(Stopped) !!!")
    print("*"*80)


  
##########################################################
#=================================
# Code for reoder character
#=================================  

class Char:
    character = None
    t = None
    l = None
    r = None
    b = None
    x_center = None
    y_center = None

# to get center of box
def getCenterPoint(l, r , t, b) :
    x = (l+r)/2
    y = (t+b)/2
    # print("Center: ("+str(x)+","+str(y)+"), ")
    return x,y

class Node:
    def __init__(self, char_data = Char, next=None):
        self.char_data = char_data
        self.next = next

class ChildLinkedList:
  def __init__(self):  
    self.head = None

  def printLL(self):
      if self.head is None:
          print("Empty List")
      else:
          n = self.head
          while n is not None:
              print(n.char_data.character ,end="")
              n = n.next
      print()
  
  def getTextLine(self):
    if self.head is None:
        pass
    else:
        n = self.head
        text = ""
        prev = n
        while n is not None:
            if ( (n.char_data.l - prev.char_data.r) > ((prev.char_data.b - prev.char_data.t)/4 )  ):
                if (prev.char_data.character in ['่','้','๊','๋','็','ิ','ี','ึ','ื','ั','์','ุ','ู','ํ','ฺ'] ):
                    if ( (n.char_data.l - prev.char_data.r) > ((prev.char_data.b - prev.char_data.t)/2) ):
                        text += " "
                else:
                    text += " "
            
            text += n.char_data.character
            prev = n
            n = n.next
        text+="\n"

        return text
        

  def insert(self, data=Char):
    newNode = Node(data)
    current = self.head
    if(current):
        prev = None
        while(current):
            if (newNode.char_data.l < current.char_data.l):
                if (self.head is current):
                    self.head = newNode
                    newNode.next = current
                    break
                newNode.next = current
                prev.next = newNode
                break
            if (current.next is None):
                current.next = newNode
                break
            prev = current
            current = current.next
    else:
        self.head = newNode

class HeadNode:
    def __init__(self, next=None, child=ChildLinkedList):
        self.line_index = None
        self.max_top = None
        self.max_bottom = None
        self.childList = child
        self.next = next

class LinkedList:
  def __init__(self):  
    self.head = None

  def getHead(self):
      return self.head
  
  def getText(self):
    if self.head is None:
        print("Empty List")
    else:
        n = self.head
        text = ""
        while n is not None:
            text += n.childList.getTextLine()
            n = n.next
        return text

  def printLL(self):
      if self.head is None:
          pass
      else:
          n = self.head
          while n is not None:
              n.childList.printLL()
              n = n.next

  def insert(self, data):
    if (self.head):
        current = self.head
        newHeadNode = HeadNode()
        newHeadNode.max_top = data.t
        newHeadNode.max_bottom = data.b
        i=2
        prev = None
        childlist = None

        while(current):
            # print( str(current.max_top) + "<=" + str(data.y_center) + "<=" + str(current.max_bottom) )
            center = data.y_center
            # -=-=-=-=-=-=-=-= Exception for THAI -=-=-=-=-=-=-=-=-=-=-=-
            # Upper and top character in thai
            if (data.character in ['่','้','๊','๋','็','ิ','ี','ึ','ื','ั','์','ํ']):
                if (data.character in ['ื','็','ี','ึ','ิ']):
                    data.l += (data.r - data.l)/2
                center = data.y_center + ((data.b - data.t)*2)
            # Lower character in thai
            if (data.character in ['ุ','ู','ฺ']):
                center = data.y_center - ((data.b - data.t)*2)
            # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
            if(current.max_top <= center <= current.max_bottom):
                childlist = current.childList
                childlist.insert(data)
                break
            prev = current
            current = current.next
            if(current is None):
                childlist = ChildLinkedList()
                childlist.insert(data)
                newHeadNode.childList = childlist
                newHeadNode.line_index = i
                prev.next = newHeadNode
            i+=1

    else:
        self.head = HeadNode()
        current = self.head
        current.childList = ChildLinkedList()
        current.line_index = 1
        current.max_top = data.t
        current.max_bottom = data.b
        current.childList.insert(data)     

  def sort(self):
      newlist = []
      if self.head is not None:
          current = self.head
          while current:
              newlist.append(current)
              current = current.next
          newlist = sorted(newlist, key= lambda HeadNode: HeadNode.max_top)
    #   print("After: ")
    #   for node in newlist:
    #       node.childList.printLL()
      return newlist
##########################################################

#=================================
# main
#=================================        
if __name__ == "__main__":
    # Check Parameters
    print(cv2.__version__)
    if len(sys.argv) > 3  :   
        get_screen_resolution()
        # Check if Training Mode
        if sys.argv[1] == 'train' :
            dataYaml = sys.argv[2]; configYaml=sys.argv[3]; epochNo = int(sys.argv[4])
            train_custom_data(dataYaml, configYaml, epochNo)   
        elif sys.argv[1] == 'predict' :
            bestWeight = sys.argv[2];   dataFolder = sys.argv[3];   
            showMode = sys.argv[4]; saveMode = sys.argv[5]
            predict_data(bestWeight, dataFolder, showMode, saveMode)   
        else :
            show_program_usage() 
    else :
        show_program_usage()            