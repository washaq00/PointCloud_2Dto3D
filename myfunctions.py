import matplotlib.pyplot as plt
from torch import permute

def print_image(img):
    img = permute(img, (1,2,0))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.show()