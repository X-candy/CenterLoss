import caffe
import numpy as np
import lmdb
from caffe.proto import caffe_pb2
import matplotlib.pyplot as plt
import sys
import cv2

BATCH=100


def draw(path):
    with open(path,'r') as f:
        lines = f.readlines()

    d = dict()

    for line in lines:
        sps = line.split()
        label = int(sps[0])
        if label not in d:
            d[label] = []
        pt = [float(sps[1]), float(sps[2])]
        d[label].append(pt)
    print len(d)
    color = ['blue','green','red','cyan','magenta',
        'yellow','black','white','gray','pink']
    for key in d:
        print str(key) + "\t" + str(len(d[key]))
        val = np.array(d[key])
        x = val[:,0]
        y = val[:,1]
        plt.scatter(x,y, label=key, c = color[key])

    plt.legend()
    plt.show()



if __name__ == '__main__':
    model_file = "/Users/xcandy/CaffeWorkSpace/Mobile-NCNN/mobilenet_deploy.prototxt"
    weight_file = "/Users/xcandy/CaffeWorkSpace/Mobile-NCNN/mobileNet_iter_20000.caffemodel"
    lmdb_file ="/Users/xcandy/CaffeWorkSpace/Mobile-NCNN/mnist_test_lmdb"
    n = caffe.Net(model_file, weight_file, caffe.TEST)
    input_name = n.inputs[0]
    transformer = caffe.io.Transformer({input_name: n.blobs[input_name].data.shape});
    transformer.set_input_scale(input_name, 255.0)

    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe_pb2.Datum()

    count = 0
    batch_data = []
    batch_label = []

    fo = open("center_feat.log", "w")
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        im = data.astype(np.uint8)
        img = np.frombuffer(data,dtype=np.uint8).reshape(28,28,1)
        # cv2.imshow("img",img)
        # cv2.waitKey(1)
        count += 1
        batch_data.append(im)
        batch_label.append(label)

        if count % 64 == 0:
            data = np.array(batch_data, dtype=np.float32) / 255.0
            n.forward_all(**{input_name: data})
            res_data = n.blobs['fc1'].data
            predict_data = n.blobs['fc2'].data
            # for predict_score, label in zip(predict_data, batch_label):
            #     print str(label) + '\t'+str(predict_score)

            for res, label in zip(res_data, batch_label):
                featStr = [str(f) for f in res]
                str_line = str(label) + '\t' + '\t'.join(featStr)+'\n'
                #print str_line
                fo.writelines(str_line)
            batch_data = []
            batch_label = []

    fo.close()
    draw('center_feat.log')

