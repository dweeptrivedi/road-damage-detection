
yolo_weights_list = [20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000]
thresh_list = [0.01,0.05,0.1,0.15,0.2]
#yolo_weights_list = [55000]
#thresh_list = [0.15]
#nms_list = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]
#nms_list = [0.75,0.8,0.85,0.90,0.95]
nms_list = [0.96,0.97,0.98,0.99]
#nms_list = [0.991, 0.992, 0.993,0.994,0.995,0.996,0.997,0.998,0.999,1.0,1.1]

caffe_weights_list = [40000,60000,80000]
classifier_list = [0,1]
approach_list = ["one-phase","two-phase","augmented","augmented2","cropped"]



thresh_list = [0.1,0.15]
yolo_weights_list = [50000]
nms_list = [0.99]
classifier_list = [0]

with open("exec.sh","w") as f:
    for app in approach_list:
        for y in yolo_weights_list:
            for t in thresh_list:
                for nms in nms_list:
                    if app=="two-phase":
                        for c in caffe_weights_list:
                            for s in classifier_list:
                                f.write("python3 detector.py --approach={} --yolo={} --caffe={} --thresh={} --nms={} --scaled={}\n".format(app,y,c,t,nms,s))
                    else:
                        f.write("python3 detector.py --approach={} --yolo={} --thresh={} --nms={}\n".format(app,y,t,nms))

