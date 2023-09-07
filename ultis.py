from imutils.contours import sort_contours
import numpy as np
import pytesseract
import imutils
import sys
import cv2


def detect_barcode(image, show=False):
    image = cv2.resize(image, (800, 500))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (H, W) = gray.shape
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    # smooth the image using a 3x3 Gaussian blur and then apply a
    # blackhat morpholigical operator to find dark regions on a light
    # background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
    # cv2.imshow("Blackhat", blackhat)
    cv2.imshow("original", image)
    ################################

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.absolute(grad)
    (minVal, maxVal) = (np.min(grad), np.max(grad))
    grad = (grad - minVal) / (maxVal - minVal)
    grad = (grad * 255).astype("uint8")
    # cv2.imshow("Gradient", grad)
    #################

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(grad, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Rect Close", thresh)


    ################################################################
    # find contours in the thresholded image and sort them from bottom
    # to top (since the MRZ will always be at the bottom of the passport)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = sort_contours(cnts, method="bottom-to-top")[0] #because the barcode of student card is nearest the bottom
    # initialize the bounding box associated with the MRZ
    mrzBox = None

    # loop over the contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ratioW= w/W
        ratioH = h/H
        # print("################################")
        # print(x, y, w, h)
        # print(ratioW, ratioH)
        # print("################################")

        # mrz = image[y:y + h, x:x + w]
        # cv2.imshow(str(x)+", "+str(y)+", "+str(w)+", "+str(h), mrz)
        
        if ratioW > 0.36 and ratioH > 0.08 and ratioW<0.5 and ratioH<0.3:
            mrzBox = (x, y, w, h)
            mrz = image[y:y + h, x:x + w]
            cv2.imshow(str(x)+", "+str(y)+", "+str(w)+", "+str(h), mrz)
            break
        
    #####################################
    # if mrzBox is None:
    # 	print("[INFO] MRZ could not be found")
    # sys.exit(0)

    # pad the bounding box since we applied erosions and now need to
    # re-grow it
    if mrzBox is None:
        print("MRZ region not detected.")
    else:
        (x, y, w, h) = mrzBox
    pX = int((x + w) * 0.03)
    pY = int((y + h) * 0.03)
    (x, y) = (x - pX, y - pY)
    (w, h) = (w + (pX * 2), h + (pY * 2))
    # extract the padded MRZ from the image
    mrz = image[y:y + h, x:x + w]
    new_width = int(w * 1.5)  # Adjust the scaling factor as needed
    new_height = int(h * 1.5)  # Adjust the scaling factor as needed
    resized_mrz = cv2.resize(mrz, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Display or save the resized MRZ image
    if show:
        cv2.imshow('Resized MRZ', resized_mrz)
        # cv2.imshow("mrz", mrz)
        
    return resized_mrz

########################################################
img = cv2.imread("images\\full_cap.jpg")
img = detect_barcode(img, show=True)
# cv2.imshow("xxx", img)
cv2.waitKey(3000)



