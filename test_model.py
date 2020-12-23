import argparse
import time
import datetime
import numpy as np
from pathlib import Path
import cv2
import keras
from model import get_model
from noise_model import get_noise_model


def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="grid,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    # args = parser.parse_args()
    # args = parser.parse_args(
    #                         ["--image_dir", "dataset/DCM_file",
    #                          "--weight_file", "clean/weights.060-6.305-40.87403.hdf5",
    #                         "--output_dir","TestOutput"
    #                         ])
    
    args = parser.parse_args(
                            ["--image_dir", "dataset/Set14",
                              "--weight_file", "grid/weights.056-6.083-41.60287.hdf5",
                            "--output_dir","TestOutput"
                            ])
    
    return args


def get_image(image):
    # image = np.clip(image, 0, 255)
    # return image.astype(dtype=np.uint8)
    return image.astype(dtype=np.uint16)


def resize_img(image,rate = 2):
    # reImageIn = cv2.imread(image)
    reImageIn = image
    imgH , imgW = reImageIn.shape[:2]
    resizeImgH = int(rate*imgH)
    resizeImgW = int(rate*imgW)
    
    resizeImgOut = cv2.resize(reImageIn,(resizeImgH,resizeImgW),interpolation=cv2.INTER_CUBIC)
    
    return resizeImgOut

def main():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    
    val_noise_model = get_noise_model(args.test_noise_model)
    model = get_model(args.model)
    model.load_weights(weight_file)
    
    isDCMfile = True
    
    # keras.backend.set_learning_phase (0)

    if args.output_dir:
        tempOutputDir = args.output_dir
        nowDate= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tempOutputDir = tempOutputDir+"/"+nowDate

        output_dir = Path(tempOutputDir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print("OutputDir:: " + str(output_dir))

    image_paths = list(Path(image_dir).glob("*.*"))
    
    tokens = args.test_noise_model.split(sep=",")
    noisetype  = tokens[0]
    
    
    if isDCMfile == True:
        image_dir = "dataset/DCM_file"
        image_paths = list(Path(image_dir).glob("*.*"))
        
        for image_path in image_paths:
            # imageIn = cv2.imread(str(image_path))
            imageIn = np.fromfile(str(image_path),dtype=np.uint16).reshape(2800,2304,1)
            h_o, w_o = imageIn.shape[:2]
            image_size = 512 # display size 
            
            for idx in range(10):            
                i = np.random.randint(h_o - image_size + 1)
                j = np.random.randint(w_o - image_size + 1)
                
                image = imageIn[i:i + image_size, j:j + image_size]
                # image = np.expand_dims(image,-1).astype(np.uint16)
                
                h,w = image.shape[:2]
                
                image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
                
                h,w = image.shape[:2]
                print("size : %d , %d " %(h,w) )
                
                out_image = np.zeros((h, w * 3, 1), dtype=np.uint16)
                
                # noise_image = val_noise_model(image)
                noise_image = image
                t0 = time.time()
                pred = model.predict(np.expand_dims(noise_image, 0))
                t1 = time.time()
                print("process time: ",(t1-t0))
                denoised_image = get_image(pred[0])
                out_image[:, :w] = image
                out_image[:, w:w * 2] = noise_image
                out_image[:, w * 2:] = denoised_image
        
                if args.output_dir:
                    cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
                    print("output : ", image_path.name,".png")
                else:
                    cv2.imshow("result", out_image)
                    key = cv2.waitKey(-1)
                    # "q": quit
                    if key == 113:
                        return 0

    else:
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            
            if noisetype   == "grid":
                imageorg = image.copy()
                image=np.expand_dims(cv2.cvtColor(imageorg ,cv2.COLOR_BGR2GRAY), -1)    
                image = resize_img(image,1)      
                image = np.expand_dims(image,-1).astype(np.uint16)
    
            else:
                image = resize_img(image,1)      
            
    
            h, w, _ = image.shape
            image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            h, w, _ = image.shape
            print("processImage: ", image_path)
            print("size : %d , %d " %(h,w) )
    
            if noisetype   == "grid":
                out_image = np.zeros((h, w * 3, 1), dtype=np.uint16)
            else:
                out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
            noise_image = val_noise_model(image)
            t0 = time.time()
            pred = model.predict(np.expand_dims(noise_image, 0))
            t1 = time.time()
            print("process time: ",(t1-t0))
            denoised_image = get_image(pred[0])
            out_image[:, :w] = image
            out_image[:, w:w * 2] = noise_image
            out_image[:, w * 2:] = denoised_image
    
            if args.output_dir:
                cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)
                print("output : ", image_path.name,".png")
            else:
                cv2.imshow("result", out_image)
                key = cv2.waitKey(-1)
                # "q": quit
                if key == 113:
                    return 0



if __name__ == '__main__':
    main()
