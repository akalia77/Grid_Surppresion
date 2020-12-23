import argparse
import string
import random
import numpy as np
import cv2
from Test_grid import getRate
from Test_grid import createImageWidthGrid

def get_noise_model(noise_type="gaussian,0,50"):
    tokens = noise_type.split(sep=",")

    if tokens[0] == "gaussian":
        min_stddev = int(tokens[1])
        max_stddev = int(tokens[2])

        def gaussian_noise(img):
            noise_img = img.astype(np.float)
            stddev = np.random.uniform(min_stddev, max_stddev)
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
            return noise_img
        return gaussian_noise
    elif tokens[0] == "clean":
        return lambda img: img
    elif tokens[0] == "text":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_text(img):
            img = img.copy()
            h, w, _ = img.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_for_cnt = np.zeros((h, w), np.uint8)
            occupancy = np.random.uniform(min_occupancy, max_occupancy)

            while True:
                n = random.randint(5, 10)
                random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
                font_scale = np.random.uniform(0.5, 1)
                thickness = random.randint(1, 3)
                (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
                x = random.randint(0, max(0, w - 1 - fw))
                y = random.randint(fh, h - 1 - baseline)
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
                cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

                if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
                    break
            return img
        return add_text
    elif tokens[0] == "impulse":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_impulse_noise(img):
            occupancy = np.random.uniform(min_occupancy, max_occupancy)
            mask = np.random.binomial(size=img.shape, n=1, p=occupancy / 100)
            noise = np.random.randint(256, size=img.shape)
            img = img * (1 - mask) + noise * mask
            return img.astype(np.uint8)
        return add_impulse_noise
    elif tokens[0] == "grid":
        min_occupancy = int(tokens[1])
        max_occupancy = int(tokens[2])

        def add_Grid_noise(img):
            shapesize = len(img.shape)

            if shapesize ==2 :
                print("gen noise grid :: input image shape is 2")            
            h,w,_ = img.shape

            angle_val = np.random.uniform(0, 1)
            isHorizontal = np.random.randint(0,2) 


            # t1_rate= 2 * np.random.binomial(10,0.5)/10
            # t2_rate = -3* np.random.binomial(10,0.7)/10

            t1_rate= 2 * np.random.binomial(10,0.5)/10
            t2_rate = -4* np.random.binomial(10,0.7)/10


            t1_with = 1*np.random.uniform(0.7, 5)
            tw_total = t1_with + np.random.uniform(0.3, 2)
            in_type = img.dtype
            inImg = img.copy()

            retImg = createImageWidthGrid(inImg ,w,h,
                                    t1_rate,t2_rate,t1_with,tw_total,
                                    angle_val,isHorizontal,0)
            
            retImg = retImg.astype(in_type)

            return retImg
        
        return add_Grid_noise
    else:
        raise ValueError("noise_type should be 'gaussian', 'clean', 'text', or 'impulse'")


def get_args():
    parser = argparse.ArgumentParser(description="test noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_size", type=int, default=256,
                        help="training patch size")
    parser.add_argument("--noise_model", type=str, default="gaussian,0,50",
                        help="noise model to be tested")

    
    # args = parser.parse_args()
    args = parser.parse_args(["--noise_model", "grid,0,95"])
    return args


def main():
    
    args = get_args()
    image_size = args.image_size
    noise_model = get_noise_model(args.noise_model)

    tokens = args.noise_model.split(sep=",")
    noisetype  = tokens[0]

    inFileName  = "./dataset/Set14/barbara.bmp"

    img=cv2.imread(inFileName)

    imgGray =np.expand_dims(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), -1)


    while True:
        if noisetype == 'grid':
            image = imgGray.copy()
            h,w = image.shape[:2]
            out_image = np.zeros((h, w * 2, 1), dtype=image.dtype)
            
            # image = np.ones((image_size, image_size, 1), dtype=np.uint8) * 128
        else:    
            image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128
            h,w = image.shape[:2]
            out_image = np.zeros((h, w * 2, 1), dtype=image.dtype)

        noisy_image = noise_model(image)
        out_image[:,:w]=image
        out_image[:,w:w*2] =noisy_image


        # cv2.imshow("noise image", noisy_image)
        cv2.imshow("noise image", out_image)
        key = cv2.waitKey(-1)

        # "q": quit
        if key == 113:
            return 0


if __name__ == '__main__':
    main()
