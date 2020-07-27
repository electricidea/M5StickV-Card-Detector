import image
import lcd
import sensor
import sys
import time
import KPU as kpu
from fpioa_manager import *

import KPU as kpu

lcd.init()
lcd.rotation(2)

try:
    from pmu import axp192
    pmu = axp192()
    # reduce screen brightness
    pmu.setScreenBrightness(8) #[7..15] works
    pmu.enablePMICSleepMode(True)
except:
    pass

try:
    img = image.Image("/sd/startup.jpg")
    lcd.display(img)
except:
    lcd.draw_string(lcd.width()//2-100,lcd.height()//2-4, "Error: Cannot find start.jpg", lcd.WHITE, lcd.RED)

# load the trained k-model
# the model was trained with then M5Stack V-Traing service:
# http://v-training.m5stack.com/
task = kpu.load("/sd/card_03.kmodel")

# English:
labels=["9","8","7","10","ACE","JACK","QUEEN","KING"]
# German:
#labels=["9","8","7","10","Ass","Bube","Dame","Koenig"]

sensor.reset()
# initialize color images
sensor.set_pixformat(sensor.RGB565)
# The memory can't analyze models with resolution higher than QVGA
# and the model is trained with QVGA too
sensor.set_framesize(sensor.QVGA) #QVGA=320x240

# It is a good idea to tune the camera exposure
# and disable the auto-gain function.
# Please tune this settings to get best picture quality
sensor.set_auto_exposure(False, exposure_us=125) # 500
sensor.set_auto_gain(False)     # must turn this off to prevent image washout...
sensor.set_auto_whitebal(True)  # turn this off for color tracking

# only use a squared window
sensor.set_windowing((224, 224))
sensor.run(1)

lcd.clear()


while(True):
    # get an image snapshot
    img = sensor.snapshot()
    # let the model find something...
    fmap = kpu.forward(task, img)
    plist=fmap[:]
    # get the best fitting result
    pmax=max(plist)
    # any value lower than 0.96 is "not accurate detected"
    if pmax > 0.95:
        # get the corresponding index for the label
        max_index=plist.index(pmax)
        # print the label on the image
        img.draw_string(0, 0, "%s"%(labels[max_index].strip()),color=(0,255,0),scale=5)
    # scale sensor image to display size
    img2 = img.resize(135,135) # LCD: 135*240
    a = lcd.display(img2)

a = kpu.deinit(task)
