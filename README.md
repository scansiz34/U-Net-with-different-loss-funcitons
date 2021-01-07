# Effects of Loss Functions on Gland Segmentation in Colon Histology Images

The primary focus of the project is to compare the performance of some of the loss functions based on their objectives and propose an approach for their usage on the dataset of stained histopathology glands.

Gland segmentation is a challenging task because, glands are often in close proximity and they vary in their morphological appearance.

U-Net model is considered as sub-optimal approach for this task because it has down-sampling layers that results in loss of spatial information and poor segmentation.

The choice of loss function is important because it helps the model to perform better segmentations. So, inabilities of the models could be tackled by developing different loss functions. 

The dataset is available at servers of Department of Computer Enginnering at Bilkent University.

Path of images: "/media/hdd3/gunduz/cansari/datasets/tubule/images/" 

Annotations: "/media/hdd3/gunduz/cansari/datasets/tubule/gold_standard_segm"
