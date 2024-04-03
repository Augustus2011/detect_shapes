
## huge credits to Johannes Rieke ##https://github.com/jrieke/shape-detection/blob/master/color-multiple-shapes.ipynb

#manage file,path,write,load
import os
import json
import glob
import shutil

#manage images,bounding boxes
import cairo
import numpy as np
import cv2

#other
from utils import check_dependencies,set_seed
import tqdm


def generate_and_save(num_imgs: int, img_size: int = 300, min_object_size: int = 7, max_object_size: int = 150, num_objects: int = 1, json_name: str = "num_objects1.json")-> None:
    
    path_ = f"/Users/kunkerdthaisong/ipu/intern/generated_img/numObs{num_objects}/"#Change to your dir
    if not os.path.exists(path_):
        os.makedirs(path_)

    bboxes = np.zeros((num_imgs, num_objects, 4)) # (num_imgs,num_objects,[x1,y1,x2,y2]..) x,y bottom left to top right
    imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  
    shapes = np.zeros((num_imgs, num_objects), dtype=int)
    num_shapes = 3

    for i_img in tqdm.tqdm(range(num_imgs)):
        surface = cairo.ImageSurface.create_for_data(imgs[i_img], cairo.FORMAT_ARGB32, img_size, img_size)
        cr = cairo.Context(surface)
        cr.set_source_rgb(1, 1, 1)
        cr.paint()

        for i_object in range(num_objects):
            shape = np.random.randint(num_shapes)
            shapes[i_img, i_object] = shape
            if shape == 0:  # Square
                w,h = np.random.randint(min_object_size, max_object_size,size=2)
                w = max(1, min(w, img_size))  # Ensure width is within the (300,300) and > 0
                h = w                                       # Square
                x = np.random.randint(1, img_size - w + 1)  # Ensure x is within (300,300) and not out of bounding box when add width
                y = np.random.randint(1, img_size - h + 1) 
                bboxes[i_img, i_object] = [x, y, w, h]
                cr.rectangle(x, y, w, h) # Draw rectangle

            elif shape == 1:  # Circle   
                r = 0.5 * np.random.randint(min_object_size, max_object_size)
                r = max(1, min(r, min(img_size, img_size) / 2))  # Ensure radius is < 300 and > 0
                x = np.random.randint(r, img_size - r)  # Ensure x is within (r, img_size - r)
                y = np.random.randint(r, img_size - r)  # Ensure y is within (r, img_size - r)
                bboxes[i_img, i_object] = [x - r, y - r, 2 * r, 2 * r]
                cr.arc(x, y, r, 0, 2*np.pi) # Draw circle

            elif shape == 2:  # Triangle
                w,h = np.random.randint(min_object_size, max_object_size, size=2)
                w = max(1, min(w, img_size))            # Ensure w is within (300,300)
                h= w                                    # From evaluate set that triangle is isosceles triangle and EDA show that base(w)==height(h)
                x = np.random.randint(w, img_size - w)
                y = np.random.randint(h, img_size - h)  
                bboxes[i_img, i_object] = [x, y-h, w,h] 
                cr.move_to(x, y)                        # Draw triangle
                cr.line_to(x + w, y)
                cr.line_to(x + (w / 2), y - h)          # draw stand triangle
                cr.line_to(x, y)
                cr.close_path()

            cr.set_source_rgb(0, 0, 0)                  # Set shapes to black color
            cr.fill()

    imgs_rgb = imgs[..., 2::-1]  # Convert BGRA to RGBA
    print("gen images:",imgs.shape,"bboxes:",bboxes.shape,"shapes:",shapes.shape)

    # Save images and bounding box data
    for i in range(num_imgs):
        name = f"numObs{num_objects}_{i}.png"
        cv2.imwrite(os.path.join(path_, name), cv2.bitwise_not(imgs_rgb[i])) # Convert black to white and white to vlck
        if i == 0:
            dict_ = {os.path.join(path_, name): {"bboxes": bboxes[i].tolist(), "shapes": shapes[i].tolist()}}
        else:
            dict2 = {os.path.join(path_, name): {"bboxes": bboxes[i].tolist(), "shapes": shapes[i].tolist()}}
            dict_.update(dict2)
    
    # Save to json format
    with open(json_name, "w") as outfile:
        json.dump(dict_, outfile)

#turn to yolo format
def yolo_format(Class:int,CX:float,CY:float,W:float,H:float)-> str:
    txt = str(Class) + ' ' + str(CX) + ' ' + str(CY) + ' ' + str(W) + ' ' + str(H)
    return txt

#create .txt for feed into yolo model 
def labels_txt(cl1:dict,path:str="/Users/kunkerdthaisong/ipu/intern/train/labels",name:str=None)-> None:
    lines=[]
    name=name+'.txt'
    for bbox, shape in zip(cl1['bboxes'], cl1['shapes']): #cl1 is dict {"bboxes":[[x1,y1,x2,y2],[...]],"shapes":[0,...]} #shapes 0:is square ,1: is circle and 2: is triangle
        x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
        CX = (x1+(x2/2)) /300
        CY = (y1+(y2/2)) /300
        W = (x2)/300
        H = (y2)/300
        print(shape,CX,CY,W,H,x1,x2,y1,y2)
        lines.append(yolo_format(shape,CX,CY,W,H))

    # save file .txt
    with open(f"{path}/{name}","w") as f:
        f.write('\n'.join(lines))

def create_train_val_folders(mode:str="train")-> None:

    if mode=="train": #create folders
        
        if not os.path.exists("train"):
            os.makedirs("train")
        
        if not os.path.exists("train/images"):
            os.makedirs("train/images")

        if not os.path.exists("train/labels"):
            os.makedirs("train/labels")

        
        paths=glob.glob('/Users/kunkerdthaisong/ipu/intern/generated_img/numObs1/*.png')[:900] # Select images and make labels .txt
        for i in range(2,6):
            paths.extend(glob.glob(f'/Users/kunkerdthaisong/ipu/intern/generated_img/numObs{i}/*.png')[:900])
        
        for p in paths:
            path_save='/Users/kunkerdthaisong/ipu/intern/train/images/'
            name=os.path.basename(p).split(".png")[0]
            shutil.copy(p,path_save+f"{name}.png")
            cl1=data.get(p)
            labels_txt(cl1=cl1,name=name,path="/Users/kunkerdthaisong/ipu/intern/train/labels")

    elif mode=="val":
        if not os.path.exists("val"):
            os.makedirs("val")

        if not os.path.exists("val/images"):
            os.makedirs("val/images")

        if not os.path.exists("train/labels"):
            os.makedirs("val/labels")
        
        paths=glob.glob('/Users/kunkerdthaisong/ipu/intern/generated_img/numObs1/*.png')[900:]
        for i in range(2,6):
            paths.extend(glob.glob(f'/Users/kunkerdthaisong/ipu/intern/generated_img/numObs{i}/*.png')[900:])
        
        for p in paths:
            path_save='/Users/kunkerdthaisong/ipu/intern/val/images/'
            name=os.path.basename(p).split(".png")[0]
            shutil.copy(p,path_save+f"{name}.png")
            cl1=data.get(p)
            labels_txt(cl1=cl1,name=name,path="/Users/kunkerdthaisong/ipu/intern/val/labels")




if __name__=="__main__":
    check_dependencies()
    
    seed=set_seed(42)
    seed.set()
    
    # num_imgs==2000 from methodology ,img_size from evaluate set ,min_object_size from EDA ,max_object_size from EDA, num_object is number of shapes in an image
    # Balancing num_objects 1 to 5 shape in an image
    data1=None
    data2=None
    data3=None
    data4=None
    data5=None
    for i in range(1,6): # i in num_objects 
        generate_and_save(num_imgs=1000,img_size=300,min_object_size=7,max_object_size=150,num_objects=i,json_name=f"numObjects{i}.json") 

        with open(f"/Users/kunkerdthaisong/ipu/intern/numObjects{i}.json") as json_file:# Merge five json  or one by one ,In this task i would like to train all of them
            if i==1:
                data1=json.load(json_file)
                json_file.close()
            elif i==2:
                data2=json.load(json_file)
                json_file.close()
            elif i==3:
                data3=json.load(json_file)
                json_file.close()
            elif i==4:
                data4=json.load(json_file)
                json_file.close()
            elif i==5:
                data5=json.load(json_file)
                json_file.close()

    merged={**data1,**data2,**data3,**data4,**data5}

    #save json
    with open("/Users/kunkerdthaisong/ipu/intern/merged.json", "w") as outfile:
        json.dump(merged, outfile)
    
    #load our merged.json
    with open("/Users/kunkerdthaisong/ipu/intern/merged.json","r") as json_file:
        data=json.load(json_file)

    #make train set
    create_train_val_folders(mode="train")# "tran","val"
    create_train_val_folders(mode="val")