# -*- coding: utf-8 -*-
"""
 @File    : rest.py
 @Time    : 2020/4/6 上午9:39
 @Author  : yizuotian
 @Description    : restful服务
"""

import sys,os
import cv2
import crnn

from craft_text_detector import Craft



if __name__ == '__main__':
    """
    Usage: 
    export KMP_DUPLICATE_LIB_OK=TRUE
    python rest.py -l output/crnn.horizontal.061.pth -v output/crnn.vertical.090.pth -d cuda
    """


    # set image path and export folder directory
    image = sys.argv[1] # can be filepath, PIL image or numpy array
    output_dir = 'outputs/'

    # create a craft instance
    craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image)
    boxes = prediction_result['boxes']
    image = cv2.imread(image)
    model = crnn.Predict()

    basename = sys.argv[1].split('/')[-1].split('.')[0]
    cropdir='outputs/%s_crops' % basename
    print(cropdir)

    for name in os.listdir(cropdir):
        img = cv2.imread(cropdir+'/'+name)
        cv2.imshow('show', img)

        text = model.predict(img)
        print("text:{}".format(text))
        cv2.waitKey(0)




    # 启动restful服务
    #f=open("label.pk",'wb')
    #pickle.dump(alpha,f)





