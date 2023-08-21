# Feature extraction and training some tree models(XGBoost and Random Forest)
Used 1 second windows using the channel wise annotation (.lbl) for classification of events as either `bckg` or `seizure`. Number of channels can be either `22` or `20` depending on the montage types. The input vector will have the shape of $[C,S_{freq} * T]$ and labels have a shape of $[C,]$. Where $S_{freq}$ is the sampleing frequency and $T$ is window size and $C$ is the number of channels.
## Feature Selection
For training the xgboost and random forest models, i used `7` set of features per channels:
1) The sum of the power spectral density values for `alpha`,`beta`,`delta`,`gamma`,and `theta` componets of the frequency domain.
2) The `mean` and `std` of the signals in each channel in the time domain.
This performed for each window and channel independently.
We now have $[C,7]$ input vectors and $[C,]4 label vectors for each `T` second windows.

We can match the `sensitivity(recall)` values published [here](https://sci-hub.se/https://ieeexplore.ieee.org/document/9353625), but it has poor `specifity` or false-alarm rate.

### TODO
Improve the the `specifity` of the model.