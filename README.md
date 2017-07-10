Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths
=============
This is the implement of  paper "Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths"(Donghui Yanan Huanhang+ 2017 CVPR)
[https://arxiv.org/abs/1703.05002]


1. Get  feature Data.
---
We use 3 different neural networks to extract our feature. 

        GoogleNet: pool5/7*7_s1    bvlc_googlenet.caffemodel
        VGG19: fc8 layer VGG_ILSVRC_19_layers.caffemodel
        RESNET: fc1000 ResNet-152-model.caffemodel
You can download the picture in those website.

        AwA: http://attributes.kyb.tuebingen.mpg.de/
        CUB: http://www.vision.caltech.edu/visipedia/CUB-200.html
        Dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/
        ImageNet and ImageNet21K: http://www.image-net.org/
put the extract feature matrix X in 'dataset/AwA/feature_mat/' and  named as AwA_goog, AwA_vgg, AwA_res. and the same as the CUB Dogs ImageNet datset


2.Get attribute Data.
---
We use wikipedia to train word2vec.

        glove: 300-dims
        word2vec cbow: 500-dims
        word2vec skipgram: 500-dims

and some dataset has handcraft attribute

        AwA: 85-dims
        CUB: 312-dims
put the word2vec or handcraft data matrix K in 'dataset/datasetname/knowledge_mat/' named as datasetname_w_cbow5,datasetname_w_skipgram5, datasetname_a_prob,datasetname_w_glove3

3.By the way
----
the classes order  in each dataset can found in '/dataset/datasetname/constants' or '/dataset/datasetname/classes' and you may need normalize the feature to -1 ~+1 to get the experiment result

4.Run
---
Open the Matlab, make the current dir to be the Relational-Knowledge-Transfer-for-ZSL, run "code/ZSL/v-release/CSC_main.m"

If you want U2T transductive result, change the U2T = 1 else U2T = 0.

Citation
-------------------------
If you find the dataset and toolbox useful in your research, please consider citing:
<pre>
@article{li2017zero,
  title={Zero-Shot Recognition using Dual Visual-Semantic Mapping Paths},
  author={Li, Yanan and Wang, Donghui and Hu, Huanhang and Lin, Yuetan and Zhuang, Yueting},
  journal={arXiv preprint arXiv:1703.05002},
  year={2017}
}
</pre>

Any Question?
-------------------------
Send Email to Me

ynli@zju.edu.cn