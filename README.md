# Italian_to_English_Translator
## Translate Italian to English using OpenCV, EAST, tesseract and attention mechanism 

Here we are translating Italian language to English from images present in 'text_data' folder.
This same project can also be used for Live text traslation as well with a very minimal changes.

For text detection we have used pretrained EAST_Text_detection model, tesseract for text recognition and Attenion layer for translating the recognized text.

## Below are the files and folder used.
1. CS2_custom_model.ipynb- has detail comparison between attention model and transformer model.
2. all_data_att.pkl - file has all the required parameters from trained attention model.
3. As this is a custom attention model hence we have saved the model checkpoints in 'checkpoints_att' folder.
4. frozen_east_text_detection.pb - is the pretrained EAST model.
5. attention_model.py file has all the classes required to test the attention model.
6. run_attention_from_image.py - file is used to run my model. This file will pick images from 'test_data' folder and translate them.

## Below is are two sample of the process.
<img src='https://github.com/Swarupbarua/Italian_to_English_Translator/blob/master/result.png?raw=true' width="800" height="400" />
<img src='https://github.com/Swarupbarua/Italian_to_English_Translator/blob/master/results1.png?raw=true' width="800" height="400" />

## Future work
1. Here i have used horizontal rectangles to detect angled text. Hence in case of multi lines of angled text, this technique may not work well.

## Reference
1. <a href="https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/"> East text detection </a>
2. <a href="https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/"> Text skew correction </a>
