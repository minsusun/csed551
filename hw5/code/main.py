import cv2
from hw5 import TV_deconv, wiener_deconv


def main():
    img_name_list = ["boy_statue", "hanzi", "harubang", "summerhouse"]
    for img_name in img_name_list:
        b = cv2.imread(f"../images/input/{img_name}.jpg")
        k = cv2.imread(
            f"../images/input/{img_name}_out.jpg.psf.png", cv2.IMREAD_GRAYSCALE
        )
        for tol in [0.01, 0.001, 0.0001]:
            tv = TV_deconv(b, k, tol)
            cv2.imwrite(f"../images/output/tv/{img_name}_{tol}.png", tv)
        for c in [0.1, 0.01, 0.001]:
            w = wiener_deconv(b, k, c)
            cv2.imwrite(f"../images/output/wiener/{img_name}_{c}.png", w)


if __name__ == "__main__":
    main()
