from emo_gen_recon import Emo_gen_recon
import cv2

img=cv2.imread('t.JPG',cv2.IMREAD_COLOR)

emo_gen_recon = Emo_gen_recon()
res = emo_gen_recon.det_emo_gen_reco(img)
res1 = emo_gen_recon.det_emo_gen_reco(img)
print ('test~~~~~~~~~~~~~~')
print (res)
print (res1)
