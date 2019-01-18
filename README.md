# STPN: Weakly Supervised Action Localization by Sparse Temporal Pooling Network
This is a re-implement of the paper Weakly Supervised Action Localization by Sparse Temporal Pooling Network, from Phuc, Google, CVPR 2018.

## I3D features:
Create a feature directory and pour in I3D features provided from Sujoy Paul, UC, Riverside [I3D features on Thumos 2014](https://emailucr-my.sharepoint.com/:f:/g/personal/sujoy_paul_email_ucr_edu/Es1zbHQY4PxKhUkdgvWHtU0BK-_yugaSjXK84kWsB0XD0w?e=I836Fl)
(Optical flow obtained in 10 fps, I3D model pre-trained on kinetics)



## Precision Updates on Thumos 2014 Challenges


**[Our baseline is not strong enough as people care more about IoU=0.5](http://xx.xx)**


| map | IoU=0.1 | 0.2| 0.3| 0.4| 0.5|
| --- | --- | --- | --- | --- | --- |
| TCAM-paper |**52.00** | **44.70** | 26.27 | **35.50** | **25.80** |
| Ours-TCAM | 40.96 | 37.65 | 26.27 | 16.98 | 10.39 |
| Ours_Margin_Loss | 47.91 | 39.08 | **28.66** | 20.14 | 13.26 |

## Other Implementation and Results
| map | IoU=0.1 | 0.2| 0.3| 0.4| 0.5|
| --- | --- | --- | --- | --- | --- |
| [Demian Zhang-PT](https://github.com/demianzhang/weakly-action-localization) |40.80 |	34.00 |	26.90 |	20.50|	14.40 |
| [JaeYoo Park-TF](https://github.com/bellos1203/STPN) | **52.10**	| 44.20 |	34.70	|26.10 |	17.70	|

## Files
*Training/Testing.ipynb*: Files for T-CAM implementation and validation.

*Training_LSTM.ipynb*: Ours margin loss model based on T-CAM. 

*Training_Gumbel_LSTM.ipynb*: In a mess and still working on it, please ignore it. :)
