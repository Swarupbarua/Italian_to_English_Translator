## Italian_to_English_Translator
## Translate Italian to English using OpenCV, EAST, tesseract and attention mechanism 

Here we are translating Italian language to English from images present in 'text_data' folder.
This same project can also be used for Live text traslation as well with a very minimal changes.

For text detection we have used pretrained EAST_Text_detection model, tesseract for text recognition and Attenion layer for translating the recognized text.

## Below is a sample of the process.
<img src='https://github.com/Swarupbarua/Live-cam-translator/blob/master/results.png?raw=true' width="800" height="400" />

## Limitation
1. Detected text with angle faces issue while translation.
2. Because of low datapoints, we achived BLEU score as 67% on our test dataset. In real time this may vary.

## Future work
1. Working on text transalation for texts detected in an angle.
