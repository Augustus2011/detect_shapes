
#model
from ultralytics import YOLO

#mange images,tensor,numpy
import torch
import pandas as pd
import numpy as np
import cv2

#manage file,path
import glob
import os

#etc
import tqdm
from utils import check_dependencies,check_return_device


class Predictor:

    def __init__(self,model:torch.nn.Module,paths:list=glob.glob("/Users/kunkerdthaisong/ipu/intern/shape_data/images/*.png"),draw:bool=False,TTA:bool=False,model_name="YOLO"):
        self.paths=paths
        self.names=[os.path.basename(i) for i in paths] # random_shapes_0.png
        self.model=model
        self.TTA=TTA
        self.draw=draw
        self.res_df=pd.DataFrame({"empty":["yo"]})
        self.big_preds=[]
        self.model_name=model_name

    def predict_one(self,img_path:str,model:torch.nn.Module,TTA:bool=False,draw:bool=False)-> np.array: # Predict one by one image
        img=cv2.imread(img_path)
        name=os.path.basename(img_path).split(".png")[0] # or just .
        if self.model_name=="YOLO":
            with torch.no_grad(): # Make sure no gradient updated
                results=model.predict(img,augment=TTA,conf=0.7)# Change this later
            
            triangle_count = 0
            circle_count = 0
            square_count = 0

            for r in results:
                for cls in r.boxes.cls.tolist():
                    if cls == 0:
                        circle_count += 1
                    elif cls == 1:
                        square_count += 1
                    else:
                        triangle_count += 1

            if self.draw:
                cv2.putText(img, f"Circle: {circle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"Square: {square_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"Triangle: {triangle_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(f"/Users/kunkerdthaisong/ipu/intern/results/two/{name}.png",img)

            return np.asarray([square_count,circle_count,triangle_count])
        
        else:#when no use yolo
            pass
    
    def predict_batch()-> None: # Write latter
        pass

    def run(self)->None:
        for p in tqdm.tqdm(self.paths):
            self.big_preds.append(self.predict_one(p,model=self.model,TTA=self.TTA,draw=self.draw)) # You can change to predict_batch later

        self.res_df=pd.DataFrame({"name":self.names,"counts":self.big_preds}) #big_preds is np.array =>[[square_count,circle_count,triangle_count],.....]



    
if __name__=="__main__":
    check_dependencies()
    device=check_return_device()    
    model=YOLO('/Users/kunkerdthaisong/ipu/intern/runs/detect/train7/weights/best.pt') #v8n  or change model later   train3/weights/best.pt' is yolov8m which is trained on colab machine
    model.to(device)
    model_name=model._get_name() #str YOLO

    p=Predictor(model=model,draw=True,TTA=True,model_name=model_name) # Use defualt paths
    p.run()
    res_df=p.res_df
    res_df =res_df.sort_values('name')
    res_df.to_csv("/Users/kunkerdthaisong/ipu/intern/results/results2.csv",index=False)
    print(res_df)
