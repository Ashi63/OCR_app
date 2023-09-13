from paddleocr import PaddleOCR, draw_ocr
import cv2

ocr = PaddleOCR(lang='en')
image_path = r'C:\Users\Alkashi\Desktop\ML_Project\Computer_Vision\OCR_paddlepaddle\image\input_image\sample1.png'
image_cv = cv2.imread(image_path)
#cv2.imshow('test_image',image_cv)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
image_height = image_cv.shape[0]
image_width = image_cv.shape[1]
output = ocr.ocr(image_path)

print("Output:\n",output)

# Initialize lists to store bounding boxes, text, and probabilities
boxes = [box[0] for box in output]
text = [text[1][0] for text in output]
probabilities = [prob[1][1] for prob in output]

print('Boxes: ',boxes)
print('Text: ', text)
print('Probabilities: ', probabilities)










  
  