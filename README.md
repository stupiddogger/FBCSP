FBCSP
====
Introduction
-------

This is implementation of filter-bank commons spatial patterns algorithm. If you want to know details of the algorithm, you can see :  
Kai K A, Zheng Y C, Zhang H, et al. Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface[C]// IEEE International Joint Conference on Neural Networks. 2008.

Dataset
-------

I use BCI competition 2003 graz_data as dataset, it contains only one subject and two kinds of motor imagination tasks. Therefore ,the implementation is simple, it doesn't take into account the multi_task and multi-subject situation. It can be poved according to the actual situation.

Choice
------
In this implentation, I use [4,8,12,16,20,24,28,32] as frequency bands to realize stage 1,  use MIBIF algorithm to realize stage 3 and use SVM as classifier in stage 4. It should be adjusted according to the actual situation.
