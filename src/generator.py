import cv2
import random
import os
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from bidi.algorithm import get_display


#  This is the image generator for generating the training data and validation data
def generate(saving_path, max=1000):
    x, y = 50, 35
    i = 0
    labels = []
    name = 0
    
    
    while (True):
        W, H = random.randint(150, 250), random.randint(20, 80)
        
        font_name = os.listdir('./Datasets/en-fonts')[random.randint(0, 20)]

        font = ImageFont.truetype(f'./Datasets/en-fonts/{font_name}', random.randint(15, 32))
        
        first_digit = random.randint(10, 99)
        second_digit = random.randint(100, 999)
        third_digit = random.randint(1000, 9999)
    
        if (random.randint(0,1) == 0):
            number = '09' + str(first_digit) + str(second_digit) + str(third_digit)
        else:
            number = '09' + str(first_digit) +' '+ str(second_digit) +' '+ str(third_digit)
        
        
        img = np.zeros((50, 200), np.uint8)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        bidi_text = get_display(number)

        draw = ImageDraw.Draw(img_pil)

        _, _, w, h = draw.textbbox((0, 5), number,  font=font)

        draw.text(((W-w)/2, (H-h)/2), bidi_text, (255), font=font)
        draw = ImageDraw.Draw(img_pil)

        img = np.array(img_pil)
        

        name += 1
        cv2.imwrite( saving_path + str(name) + '.jpg', np.array(img))
        labels.append(number.replace(' ', ''))
        
        # uncommend if you want to see the generated images
        # cv2.imshow('font_name2', (img))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if (name == max):
            break

    with open(f'./Datasets/train/labels.txt', 'w') as f:
        f.write('\n'.join(labels))


# generate train image
generate('./Datasets/train/', 15000)

# generate validation image
generate('./Datasets/validation/', 1000)