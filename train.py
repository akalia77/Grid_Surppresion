import argparse
import numpy as np
import cv2
import time
import datetime
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,Callback
from keras.optimizers import Adam

from model import get_model, PSNR, L0Loss, UpdateAnnealingParameter
from generator import NoisyImageGenerator, ValGenerator
from noise_model import get_noise_model
from Test_grid import getRate
from Test_grid import createImageWidthGrid

##  https://github.com/yu4u/noise2noise

class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, 
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, 
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=128,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,#default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--source_noise_model", type=str, default="grid,0,50",
                        help="noise model for source images")
    parser.add_argument("--target_noise_model", type=str, default="grid,0,50",
                        help="noise model for target images")
    parser.add_argument("--val_noise_model", type=str, default="grid,25,25",
                        help="noise model for validation source images")
    parser.add_argument("--model", type=str, default="unet",
                        help="model architecture ('srresnet' or 'unet')")
    args = parser.parse_args()

    args = parser.parse_args(["--image_dir","dataset/291",
                            "--test_dir", "dataset/Set14",
                            "--image_size", "128",
                            "--source_noise_model", "grid,0,95",
                            "--target_noise_model", "clean",
                            "--batch_size", "8",
                            "--lr", "0.001",
                            "--output_path", "grid",                            
                            ])



    return args


class LossAndErrorPrintingCallback(Callback):

  # def on_train_batch_end(self, batch, logs=None):
    # print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  # def on_test_batch_end(self, batch, logs=None):
    # print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))

  def on_epoch_end(self, epoch, logs=None):
    print('[test]:: epoch={}'.format(epoch))
    

    # tempOutputDir = args.output_dir
    # nowDate= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # tempOutputDir = tempOutputDir+"/"+nowDate

    output_dir =Path( "epochEnd/")
    output_dir.mkdir(parents=True, exist_ok=True)
    print ("[epochEnd] path: ",output_dir)

    fileName = "1417_103_r8_120cm_Long_Chest_1_2304x2800_uint16.raw"
    image_path = "dataset/DCM_file/" + fileName
    imageIn = np.fromfile(str(image_path),dtype=np.uint16).reshape(2800,2304,1)
    h_o, w_o = imageIn.shape[:2]
    image_size = 512 # display size 

    xpos =600
    ypos = 130

    image = imageIn[ypos:ypos + image_size, xpos:xpos + image_size]

    h,w = image.shape[:2]
                
    image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)

    h,w = image.shape[:2]
    print("size : %d , %d " %(h,w) )
   
    noise_image = image
    t0 = time.time()
    pred = self.model.predict(np.expand_dims(noise_image, 0))
    t1 = time.time()
    print("process time: ",(t1-t0))
    denoised_image = pred[0].astype(np.uint16)

    nowDate= datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    wFileName = str(output_dir)+"/"+nowDate+"_e"+str(epoch)+"_"+str(w)+"x"+str(h)+"_"
    denoised_image.tofile(wFileName+"pred.raw")
    noise_image.tofile(wFileName+"noise.raw")


    # Using the file writer, log the reshaped image.
    # fimg = check_predict_fig(model,isResize= flag_iamgeResize)
    # fimg = check_predict_fig(model,maxNormal=ismaxNormal ,isResize= flag_iamgeResize)
    
    # epochImg_name= writeImg_path + "img_"+str(epoch) + ".jpg"
    # plt.savefig(epochImg_name)
    # img =plot_to_image(fimg)
    
    
    
    # with file_writer.as_default():
    #   tf.summary.image("Training data", img, step=epoch)

    # self.model.predict()

def gridTest(inImg):
    #TestInput path 
    inImg =cv2.cvtColor(inImg,cv2.COLOR_BGR2GRAY)

    h,w = inImg.shape
    
    
    # inImg= np.full((h,w),100)
    
    
    
    angle_val = 0.9

    isHorizontal = 0


    t1_rate= 2
    t2_rate = -3
    
    
    t1_with = 2
    tw_total = 2.3


    retImg = createImageWidthGrid(inImg ,w,h,
                                  t1_rate,t2_rate,t1_with,tw_total,
                                  angle_val,isHorizontal,0)

    retImgOut = retImg.astype(np.uint16)

    retImgOut.tofile("gridTest1217-1046.raw")
    print("crate test image")

def main():
    args = get_args()
    image_dir = args.image_dir
    test_dir = args.test_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_model(args.model)

    
    # testFilePath = "./dataset/91/000t1.bmp"
    # inImg=cv2.imread(testFilePath)
    
    # gridTest(inImg)
    # gridTestImg = 
   

    if args.weight is not None:
        model.load_weights(args.weight)
        
    
    opt = Adam(lr=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()

    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])
    source_noise_model = get_noise_model(args.source_noise_model)
    target_noise_model = get_noise_model(args.target_noise_model)
    val_noise_model = get_noise_model(args.val_noise_model)

    tokens = args.source_noise_model.split(sep=",")
    noisetype  = tokens[0]

    generator = NoisyImageGenerator(image_dir, source_noise_model, target_noise_model,
                                    noisetype, 
                                    batch_size=batch_size,
                                    image_size=image_size)
    val_generator = ValGenerator(test_dir, noisetype,val_noise_model)
    output_path.mkdir(parents=True, exist_ok=True)
    myCallback = LossAndErrorPrintingCallback()
    
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) + 
    "/weights.{epoch:03d}-{val_loss:.3f}-{val_PSNR:.5f}.hdf5",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=True))
    callbacks.append(myCallback)
    

    hist = model.fit_generator(generator=generator,
                               steps_per_epoch=steps,
                               epochs=nb_epochs,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()
