# Pylaia - Code Explanation 

## Installation
If you have issues with Image Magick, follow the instructions on https://github.com/mauvilsa/imgtxtenh/issues/8

The other libraries (imgtxtenh and Kaldi)  have instructions in their repositories.

## Structure of preprocessing 
The following is going to be useful for using the code with another dataset:

- download.sh
- data/original : 
	- lines >> .png files
	- lines.txt >> labels
- data/splits/puigcerver/train: te.lst, tr.lst, va.lst contain the lines id
- prepare_images.sh
Create a folder data/imgs/lines and data/imgs/lines_h128 and the same for sentences
- /lines contains the images from data/original/lines enhanced by imgtxtenh, Skewing is also corrected using ImageMagick's convert. This tool is also used to remove all white borders from the images and leaving a fixed size of 20 pixels on the left and the right of the image.
- /lines_h218 contains images from /lines scaled to a fixed height of 128 pixels, while keeping the aspect ratio.
- prepare_texts.sh
Create folder data/lang/all/{lines,sentences,words} with a word-level and a character-level transcripts.

Example for lines

word.txt : a01-000u-00 A MOVE to stop Mr. Gaitskell from

char.txt : a01-000u-00 A @ M O V E @ t o @ s t o p @ M r . @ G a i t s k e l l @ f r o m

- Create folder data/lang/SPLIT/lines/word for the different splits 
tr.txt : a01-000u-00 A MOVE to stop Mr. Gaitskell from

- syms_ctc.txt: contains the match for each character

When we use get_item on a dataset, a dictionary is created:
	‘id’ : img_id
	‘txt’: label
	‘img’: the real image

Dataset_loader takes dataset and sampler as argument and call get_item on dataset wisely (multi-processing, etc.)

Trainer or Evaluator are Engines that have run_epoch that calls _iter_ of the dataloader and then run_iteration on the enumerate of the latter.

Engine are given ItemFeeder with a key: ‘id’,’txt’,’img’ of the get_item of the dataset which will extract relevant information in the run_iteration method before giving to the model. The rest of the process is standard training in Pytorch.


**Beware: Run setup.py install at each modification of a python file**