%% COMPREHENSIVE STUDY OF CNN REGRESSORS
%% Implementation of STN (Spatial Transformer Network) and ICSTN (Inverse Compositional Spatial Transformer Networks) in tensorlayer to predict transformation parameters from 2D images
%% Contact me for any inquiry: tajwar_aleef@etu.u-bourgogne.fr

The simple_cnn folder contains a simple implementation of using CNN to predict transformation parameters from simple 2D images

with_stn folder contains further implementations such as:
1) stn_with_addition: where the transformation parameters are simply added after every stn before the next localization network.
2) stn_with_compose: where the transformation parameters are composed after every stn before the next localization network 
3) stn_with_mridataset: where the implementation is applied on a far complex dataset that replicates a real MRI scan as excepted in a real world scenario
4) overkill_stn: this implementation is done using 3 cascaded stns and dropout layers.

All the folders contains the weights of the trained networks, so they can be used directly without needing to retrain them. Before running the code please change the directory of the folders.

Dataset and ground truth of a simple synthetic dataset of 10,000 images is available in the following link:
https://drive.google.com/open?id=1HSIdUEwIBS7Ku8gnQIg4s4JKxN50X-K9

More details about the project: https://tajwarabraraleef.github.io/publication/1/

