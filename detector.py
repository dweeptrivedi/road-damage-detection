import argparse

# import for caffe
## import caffe first, always. yolo loads a different(old) version of <forgot name> library, which is not supported by caffe
import os,sys
import time
import numpy as np
try:
    caffe_root = "/home/ubuntu/caffe-0.15.9/"
    sys.path.insert(0, caffe_root + 'python')
    os.environ['GLOG_minloglevel'] = '2' 
    import caffe
    from src.deploy_street import CaffePredictor
    caffe_support = True
except ImportError as e:
    print("caffe not installed, disable two-phase approach")
    caffe_support = False

# imports for darknet
import cv2
from collections import defaultdict
import src.darknet as dn


def deploy_func(input_file, approach, yolo_weight, caffe_weight, thresh, nms, scaled, cpu):
    """main function to deploy road damage detection model
    
    This function
    1) loads the darknet model based on the approach, architecture suppport
    2) calls appropriate functions to get predictions
    3) saves predictions in required format of the road damage detection competetion
    
    Args:
        input_file (str): contains list of image paths that needs detection.
        approach (str): name of the approach ["one-phase","cropped","augmented","augmented2","two-phase"]
        yolo_weight (int): yolo iteration number for weights
        caffe_weight (int): caffe iteration number for weights
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        scaled (int): caffe scale
        cpu (bool): True if want to run on CPU
    """
    
    yolo_model = (approach+"/darknet/yolov3.cfg.test").encode()
    yolo_weights = (approach+"/weights/yolov3_"+str(yolo_weight)+".weights").encode()
    yolo_data = (approach+"/darknet/damage.data").encode()

    dn.init_net(cpu)
    if cpu==False:
        dn.set_gpu(0)
    net = dn.load_net(yolo_model, yolo_weights, 0)
    meta = dn.load_meta(yolo_data)

    #list of test images
    with open(input_file,"r") as f:
        lines = f.readlines()

    imageList = [line.strip() for line in lines]

    #class name to int mapping
    name_to_id = {}
    with open(approach+"/darknet/damage.names","r") as f:
        names = f.readlines()
        for i in range(len(names)):
            name_to_id[names[i].strip()] = i+1

    if approach=="one-phase" or approach=="augmented" or approach=="augmented2" or approach=="cropped":
        pred_dict = deploy_func_one_phase(net, meta, imageList, name_to_id, thresh, nms)
        csvFile = open("output/sample_submission.csv"+"_"+approach+"_"+str(yolo_weight)+
                       "_"+str(thresh)+"_"+str(nms),"w")
    elif approach=="two-phase":
        if caffe_support == True:
            pred_dict = deploy_func_two_phase(net, meta, imageList, name_to_id, thresh, nms, caffe_weight, scaled)
        else:
            print("Caffe not supported on this system, can't run two-phase approach.")
            return
        csvFile = open("output/sample_submission.csv"+"_"+approach+"_"+str(yolo_weight)+
                       "_"+str(caffe_weight)+"_"+str(thresh)+"_"+str(nms)+"_"+str(scaled),"w")
    else:
        assert(0)

    for img_file in pred_dict:
        csvFile.write(os.path.basename(img_file)+",")
        for bbox in pred_dict[img_file]:
            csvFile.write(str(int(bbox[1]))+" "+str(int(bbox[3]))+" "+str(int(bbox[4]))+" "+str(int(bbox[5]))+" "+str(int(bbox[6]))+" ")
        csvFile.write("\n")
    csvFile.close()
    return pred_dict

def deploy_func_one_phase(net, meta, imageList, name_to_id, thresh, nms):
    """object detection pipeline for one-phase,augmented,augmented2,cropped approach
    
    Args:
        net (obj): YOLO darknet model
        meta (obj): YOLO darknet model metadata
        imageList (list): list of image paths to predict
        name_to_id (dict): class name to class id mapping
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        
    Returns:
        dict: dict of predicted bounding boxes
    """
    
    # pipeline: object detection
    box_id = 0
    pred_dict = defaultdict(dict)
    for idx, img_file in enumerate(imageList):
        dets = dn.detect(net, meta,img_file.encode('utf-8'), thresh=thresh, nms=nms)
        pred_box_list = []
        for bbox in dets:   
            [x,y,w,h] = bbox[2]
            #https://github.com/pjreddie/darknet/issues/243
            y = y-(h/2)
            x = x-(w/2)

            img = cv2.imread(img_file)

            y1_unscaled = int(max(0,y))
            y2_unscaled = int(min((y+h),img.shape[0]))
            x1_unscaled = int(max(0,x))
            x2_unscaled = int(min((x+w),img.shape[1]))

            crop_img = img[y1_unscaled:y2_unscaled,x1_unscaled:x2_unscaled]

            if crop_img.size > 0:
                pred_box_list.append(np.array([box_id, name_to_id[bbox[0].decode("utf-8")], bbox[1],x1_unscaled,y1_unscaled,x2_unscaled,y2_unscaled]))
                box_id += 1

        pred_dict[img_file] = np.array(pred_box_list)

    print("total images predicted:",len(pred_dict))
    return pred_dict


def deploy_func_two_phase(net, meta, imageList, name_to_id, thresh, nms, caffe_weight, scaled):
    """object detection pipeline for two-phase approach
    
    Args:
        net (obj): YOLO darknet model
        meta (obj): YOLO darknet model metadata
        imageList (list): list of image paths to predict
        name_to_id (dict): class name to class id mapping
        thresh (float): threshold value for detector
        nms (float): nms threshold value
        caffe_weight (int): caffe iteration number for weights
        scaled (int): caffe scale
                
    Returns:
        dict: dict of predicted bounding boxes
    """
    
    os.system("rm -rf ./two-phase/predictions")
    os.system("mkdir ./two-phase/predictions")

    # pipeline: object detection
    y_pred_temp = {}
    scale = 0.3 if scaled==1 else 0.0
    box_id = 0
    return_dict = defaultdict(dict)
    for idx, img_file in enumerate(imageList):
        dets = dn.detect(net, meta, img_file.encode('utf-8'), thresh=thresh, nms=nms)
        return_list = []
        for bbox in dets:   
            [x,y,w,h] = bbox[2]
            #https://github.com/pjreddie/darknet/issues/243
            y = y-(h/2)
            x = x-(w/2)

            img = cv2.imread(img_file)

            y1_unscaled = int(max(0,y))
            y2_unscaled = int(min((y+h),img.shape[0]))
            x1_unscaled = int(max(0,x))
            x2_unscaled = int(min((x+w),img.shape[1]))
            
            
            if scale > 0.0:
                y1 = max(0,int((1-scale)*y1_unscaled))
                y2 = min(int((y2_unscaled)*(1+scale)),img.shape[0])
                x1 = max(0,int((1-scale)*x1_unscaled))
                x2 = min(int((x2_unscaled)*(1+scale)),img.shape[1])
            else:
                y1 = y1_unscaled
                y2 = y2_unscaled
                x1 = x1_unscaled
                x2 = x2_unscaled

            crop_img = img[y1:y2,x1:x2]
            crop_img_name = os.path.basename(img_file).rsplit('.',1)
            crop_img_name = crop_img_name[0]+"_predicted_crop_"+str(int(box_id))+"."+crop_img_name[1]
            crop_img_file = os.path.abspath(os.path.join("./two-phase/predictions",crop_img_name))
            
            if crop_img.size > 0:
                pred = box_id
                cv2.imwrite(crop_img_file,crop_img)
                return_list.append([crop_img_file,[x1,y1,x2,y2],
                                    np.array([box_id,pred,bbox[1],x1,y1,x2,y2])])
                box_id += 1

        return_dict[img_file] = return_list


    #create test_8c.temp for level1:
    f = open("test_8c.temp","w")
    for img_file in imageList:
        if img_file not in return_dict:
            continue
        for bbox in return_dict[img_file]:
            segment_path = bbox[0]
            f.write(segment_path+" "+str(int(bbox[2][0]))+"\n")
    f.close()
    
    # Classifier
    model = "two-phase/caffe/train_val.prototxt.test"
    weights = "two-phase/caffe/weights/crop_"+str(scaled)+"/bvlc_googlenet_iter_"+str(caffe_weight)+".caffemodel"

    clf = CaffePredictor(model,weights)
    y_preds = clf.predict(box_id)
    print("total predicted boxesfrom caffe:",len(y_preds),"pred boxes from yolo:",box_id)
    assert(len(y_preds)==box_id)
    
    segment_predictions = {}
    with open("test_8c.temp","r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            segment_path = lines[i].strip().split(" ")[0]
            segment_class_id = y_preds[i]+1
            segment_predictions[segment_path] = segment_class_id


    #map the result of each segment to pred_dict:
    for img_file in imageList:
        if img_file not in return_dict:
            continue
        segment_list = return_dict[img_file]
        for idx,segment in enumerate(segment_list):
            segment[2] = list(segment[2])
            segment[2][1] = segment_predictions[segment[0]]
            #to keep the return format same as one-phase
            return_dict[img_file][idx] = segment[2]

    return return_dict


def main():
    global classifier_list, yolo_weights_list, caffe_weights_list, thresh_list
    
    
    parser = argparse.ArgumentParser(description='run phase2.')
    parser.add_argument('--approach', type=str,
                        help='name of the approach ["one-phase","cropped","augmented","augmented2","two-phase"]',
                        default='one-phase')
    parser.add_argument('--yolo', type=int, help='yolo iteration number for weights',default=45000)
    parser.add_argument('--caffe', type=int, help='caffe iteration number for weights', default=60000)
    parser.add_argument('--nms', type=float, help='nms threshold value', default=0.45)
    parser.add_argument('--thresh', type=float, help='threshold value for detector', default=0.1)
    parser.add_argument('--scaled', type=int, help='caffe scale', default=0)
    parser.add_argument('--gpu', type=bool, help='want to run on GPU?', default=False)
    parser.add_argument('--input-file', type=str, help='location to the input list of test images',default='test.txt')
    args = parser.parse_args()
    
    app = args.approach
    y = args.yolo
    c = args.caffe
    t = args.thresh
    nms = args.nms
    s = args.scaled
    cpu = not args.gpu
    test_file = args.input_file
    
    t1 = time.time()
    deploy_func(test_file,app,y,c,t,nms,s,cpu)
    t2 = time.time()
    if app=="two-phase":
        with open("output/time.txt"+"_"+app+"_"+str(y)+"_"+str(c)+"_"+str(t)+"_"+str(nms)+"_"+str(s),"w") as f:
            f.write('for approach:{} yolo:{} caffe:{} thresh:{} scaled:{} time:{}\n'.format(app,y,c,t,s,t2-t1))
    else:
        with open("output/time.txt"+"_"+app+"_"+str(y)+"_"+str(t)+"_"+str(nms),"w") as f:
            f.write('for approach:{} yolo:{} thresh:{} nms:{} time:{}\n'.format(app,y,t,nms,t2-t1))

if __name__=="__main__":
    main()
