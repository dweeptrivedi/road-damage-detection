import re
import os
import sys
from collections import defaultdict

caffe_root = "/home/ubuntu/caffe-0.17.0/"
caffe_binary = "build/tools/caffe"
model_mode = "test"
model = "train_val.prototxt.test"
weights = "bvlc_googlenet_iter_60000.caffemodel"
use_gpu = True
iterations = 1
output = "zaccuracy_temp.txt"
#"test -model train_val.prototxt.test -weights bvlc_googlenet_iter_20000.caffemodel -gpu 0,1 -iterations 20940 > accuracy_temp.txt 2>&1"

sys.path.insert(0, caffe_root + 'python')
os.environ['GLOG_minloglevel'] = '2' 
import caffe


class CaffePredictor:
    def __init__(self, model1, weights1):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        caffe.set_device(1)
        self.test_net = caffe.Net(model1,weights1, caffe.TEST)
    
    def predict(self,num_test_images):
        y_pred = {}
        for i in range(num_test_images):
            probs = self.test_net.forward()['prob']
            pred = probs.argmax()
            #y_pred.append(pred)
            y_pred[i] = pred
            #print ("Predicted Class: ", pred)
        return y_pred
        

if __name__ == "__main__":
    
    if len(sys.argv)<2:
        print("Incorrect Input:")
        print ("Use: python deploy_street.py <path_to_image/path_to_txt_file_containing_image_list>")
    else:    
        count_dict = defaultdict(int)
        input_file = sys.argv[1]
        if input_file[-4:]==".txt":
            f = open(input_file,"r")
            lines = f.readlines()
            f.close()
            outputFile = open("predictions.txt","w")
            for i in range(len(lines)):
                image = lines[i].strip()
                print("processing image:",i)
                pred = caffe_predict(image)
                count_dict[pred] += 1
                outputFile.write(image+" "+str(pred)+"\n")
                print("class_count:",count_dict)
            outputFile.close()
        else:
            caffe_predict(input_file)
