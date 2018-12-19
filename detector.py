import argparse

# import for caffe
## import caffe first, always. yolo loads a different(old) version of <forgot name> library, which is not supported by caffe
import os,sys
import time
import numpy as np

# imports for darknet
import cv2
from collections import defaultdict
import src.darknet as dn


def deploy_func(input_file, approach, yolo_weight, caffe_weight, thresh, nms, scaled, cpu):
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
        csvFile = open("output/sample_submission.csv"+"_"+approach+"_"+str(yolo_weight)+"_"+str(thresh)+"_"+str(nms),"w")
    elif approach=="two-phase":
        pred_dict = deploy_func_two_phase(net, meta, imageList, name_to_id, thresh, nms, caffe_weight, scaled)
        csvFile = open("output/sample_submission.csv"+"_"+approach+"_"+str(yolo_weight)+"_"+str(caffe_weight)+"_"+str(thresh)+"_"+str(nms)+"_"+str(scaled),"w")
    else:
        assert(0)

    for img_file in pred_dict:
        csvFile.write(os.path.basename(img_file)+",")
        for bbox in pred_dict[img_file]:
            csvFile.write(str(int(bbox[1]))+" "+str(int(bbox[3]))+" "+str(int(bbox[4]))+" "+str(int(bbox[5]))+" "+str(int(bbox[6]))+" ")
        csvFile.write("\n")
    csvFile.close()

def deploy_func_one_phase(net, meta, imageList, name_to_id, thresh, nms):
    
    # pipeline: object detection
    box_id = 0
    pred_dict = defaultdict(dict)
    for img_file in imageList:
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


def main():
    global classifier_list, yolo_weights_list, caffe_weights_list, thresh_list
    
    
    parser = argparse.ArgumentParser(description='run phase2.')
    parser.add_argument('--approach', type=str, help='name of the approach ["one-phase","cropped","augmented"]',default='one-phase')
    parser.add_argument('--yolo', type=int, help='yolo iteration number for weights',default=45000)
    parser.add_argument('--nms', type=float, help='nms threshold value', default=0.45)
    parser.add_argument('--thresh', type=float, help='threshold value for detector', default=0.1)
    parser.add_argument('--gpu', type=bool, help='want to run on GPU?', default=False)
    parser.add_argument('--input-file', type=str, help='location to the input list of test images',default='test.txt')
    args = parser.parse_args()
    
    app = args.approach
    y = args.yolo
    t = args.thresh
    nms = args.nms
    cpu = not args.gpu
    test_file = args.input_file
    
    t1 = time.time()
    deploy_func(test_file,app,y,None,t,nms,None,cpu)
    t2 = time.time()
    if app=="two-phase":
        with open("output/time.txt"+"_"+app+"_"+str(y)+"_"+str(c)+"_"+str(t)+"_"+str(nms)+"_"+str(s),"w") as f:
            f.write('for approach:{} yolo:{} caffe:{} thresh:{} scaled:{} time:{}\n'.format(app,y,c,t,s,t2-t1))
    else:
        with open("output/time.txt"+"_"+app+"_"+str(y)+"_"+str(t)+"_"+str(nms),"w") as f:
            f.write('for approach:{} yolo:{} thresh:{} nms:{} time:{}\n'.format(app,y,t,nms,t2-t1))

if __name__=="__main__":
    main()
