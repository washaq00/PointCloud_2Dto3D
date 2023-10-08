import matplotlib.pyplot as plt
from torch import permute
from time import perf_counter


def print_image(img):
    img = permute(img, (1, 2, 0))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.show()


def get_time(func):
    def wrapper(*args, **kwargs):

        start = perf_counter()
        func(*args, **kwargs)
        end = perf_counter()
        total_time = round(end - start,2)

        print(f"time: {total_time:.2f} seconds")

    return wrapper
