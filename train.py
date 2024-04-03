from ultralytics import YOLO
from utils import check_dependencies,check_return_device,set_seed


class Trainer: #write latter when no use yolo
    pass


if __name__=="__main__":
    s=set_seed(42)
    s.set()
    check_dependencies()
    device=check_return_device()    
    model=YOLO('yolov8n.pt')
    model.to(device)
    results=model.train(data='data.yaml',epochs=15,imgsz=300,dropout=0.1,hsv_v=0,hsv_h=0,hsv_s=0,translate=0,scale=0,fliplr=0.4,erasing=0.5,crop_fraction=1) #eventually it will be resize to 320 by defualt
