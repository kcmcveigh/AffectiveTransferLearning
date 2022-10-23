# AffectiveTransferLearning

In this project I use transfer learning with a pretrained resnet18 model to predict how pleasant or unpleasant a persons experience of 20 second video is. As a first pass I do this with static summary images (hence resnet18) of the videos. In psychology and neuroscience a common experimental assumption is that images reliably induce a uniform affective (pleasant or unpleasant states) across all participants and presentations. Previous work has shown DNNs can reliably predict normative (average across many people) affective ratings from images, yet to my knowledge no work has examined whether DNNs can predict subjective (a single person in a single viewing) affective ratings. As a first pass I treat the affective ratings - although ordinal in nature as a classification problem. Unfortunately the data collection for this project is ongoing, therefore the data is not yet publically available so the code cannot be rerun from this repo :(.

Results for 5 participants across 3 sets of training hyper parameters are shown below. Different participants are shown in different rows, the correspond to fine tuning all parameters in the network for a 5 way classification, fine tuning the just the final layer, and finetuning just the final layer for 3 way classification task in which the two levels of positive and negative are combined (3 classes: negative, neutral, positive). Due to the limited number of samples in the datasets ~ 250 per person I used five fold cross validation.
![image](https://user-images.githubusercontent.com/46719851/197419312-3b4bb51c-a77e-4d5c-8003-68389f053ff1.png)

We see across all folds the networks memorize and achieve high accuracy on the training data suggesting they’re training correctly (green lines) and that they predict the validation data above chance (redlines are each fold, thick redline is average across five folds, thick black line is chance). The absolute classification accuracy of all networks is relatively modest. As a machine learning result this modest accuracy is disapointing however it is consistent with much psychological research suggesting that a person’s affective state is governed by far more than just the visual stimuli they’re presented with but interacts complex with current ones current context, and life history.  Future directions for this work include adding video dynamics to as well as multimodal physiological state information such as electrodermal activity or heart rate, to improve prediction accuracy.

