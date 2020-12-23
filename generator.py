
import argparse
from pathlib import Path
import random
import datetime
import numpy as np
import cv2
from noise_model import get_noise_model
from keras.utils import Sequence


class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, source_noise_model, target_noise_model, noiseType,batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.source_noise_model = source_noise_model
        self.target_noise_model = target_noise_model
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.noise_type = noiseType

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        noise_type = self.noise_type

        if noise_type == "grid":
            x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint16)
            y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint16)
        else:
            x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
            y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)

        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))

            if noise_type == "grid":
                if image.shape[2] == 3:
                    image =np.expand_dims(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), -1).astype(np.uint16)
                
            
            h, w= image.shape[:2]

            if h >= image_size and w >= image_size:
                h, w = image.shape[:2]
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                clean_patch = image[i:i + image_size, j:j + image_size]
                x[sample_id] = self.source_noise_model(clean_patch)
                y[sample_id] = self.target_noise_model(clean_patch)

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir, noiseType,val_noise_model):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.noiseType = noiseType
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            img = cv2.imread(str(image_path))
            y=img.copy()

            if noiseType == "grid":
                if y.shape[2] == 3:
                    y =np.expand_dims(cv2.cvtColor(y,cv2.COLOR_BGR2GRAY), -1).astype(np.uint16)
            
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            x = val_noise_model(y)
            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]

def get_args():
    parser = argparse.ArgumentParser(description="test generator ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--sorce_model", type=int, default=256,
    #                     help="training patch size")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--source_noise_model", type=str, default="gaussian,0,50",
                        help="source model to be tested")

    parser.add_argument("--target_noise_model", type=str, default="gaussian,0,50",
                        help="target model to be tested")

    parser.add_argument("--val_noise_model", type=str, default="gaussian,0,50",
                        help="val model to be tested")


    # args = parser.parse_args()
    args = parser.parse_args(["--image_dir","dataset/Set14",
                            "--source_noise_model", "grid,0,95",
                            "--target_noise_model", "clean",
                            "--val_noise_model", "grid,0,95"])
    return args


def main():
    args = get_args()

    image_dir = args.image_dir
    image_size = args.image_size
    batch_size = args.batch_size

    tokens = args.source_noise_model.split(sep=",")
    noisetype  = tokens[0]

    inFileName  = "./dataset/Set14/barbara.bmp"

    img=cv2.imread(inFileName)

    imgGray =np.expand_dims(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), -1)

    h,w = imgGray.shape[:2]


    source_noise_model = get_noise_model(args.source_noise_model)
    target_noise_model = get_noise_model(args.target_noise_model)
    val_noise_model = get_noise_model(args.val_noise_model)
    generator = NoisyImageGenerator(image_dir, source_noise_model, target_noise_model,
                                    noisetype ,
                                    batch_size=batch_size,
                                    image_size=image_size)
    val_generator = ValGenerator(image_dir,noisetype,val_noise_model)

    x,y=generator.__getitem__(1)
    val_x = val_generator.__getitem__(1)

    nowDate= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    wFileName = "genTest_"+nowDate +".raw"

    b,h,w,c = x.shape

    out_image = np.zeros((h*b, w * 2, 1), dtype=x.dtype)

    for idx in range(len(x)) :
        out_image[h*idx:h*(idx+1),:w]=x[idx]
        out_image[h*idx:h*(idx+1),w:w*2] =y[idx]
    
    
    out_image.tofile(wFileName)
    
    print("image Info:: w=%d,h=%d,type=%s"%(
        out_image.shape[1],
        out_image.shape[0],
        out_image.dtype))

    print("wFileName: ",wFileName)
    print("gen test done")

    #         
    # cv2.imshow("x",x)



    # while True:
    #     out_image = np.zeros((h, w * 2, 1), dtype=imgGray.dtype)



    #     cv2.imshow("noise image", out_image)
    #     key = cv2.waitKey(-1)

    #     # "q": quit

    #     if key == 113:
    #         return 0

if __name__ == '__main__':
    main()
