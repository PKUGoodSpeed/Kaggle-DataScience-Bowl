# [Toxic Comment Classification Challenge](https://www.kaggle.com/c/data-science-bowl-2018)

### Description:
#### Spot Nuclei. Speed Cures.

Imagine speeding up research for almost every disease, from lung cancer and heart disease to rare disorders. The 2018 Data Science Bowl offers our most ambitious mission yet: create an algorithm to automate nucleus detection.

We’ve all seen people suffer from diseases like cancer, heart disease, chronic obstructive pulmonary disease, Alzheimer’s, and diabetes. Many have seen their loved ones pass away. Think how many lives would be transformed if cures came faster.

By automating nucleus detection, you could help unlock cures faster—from rare disorders to the common cold. Want a snapshot about the 2018 Data Science Bowl? View this video.

#### Why nuclei?

Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

By participating, teams will work to automate the process of identifying nuclei, which will allow for more efficient drug testing, shortening the 10 years it takes for each new drug to come to market. Check out this video overview to find out more.

#### What will participants do?

Teams will create a computer model that can identify a range of nuclei across varied conditions. By observing patterns, asking questions, and building a model, participants will have a chance to push state-of-the-art technology farther.

Visit DataScienceBowl.com to: 
• Sign up to receive news about the competition
• Learn about the history of the Data Science Bowl and past competitions
• Read our latest insights on emerging analytics techniques

### Evaluation:
This competition is evaluated on the [mean average precision at different intersection over union (IoU) thresholds](https://www.kaggle.com/c/data-science-bowl-2018#evaluation).


### Submission File

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted nucleus segmentation for the given ImageId
```
ImageId,EncodedPixels  
0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,1 1  
0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,2 3 8 9  
etc...
```

### Group name candidates:

- **Phantom Brigade** 

### Group Rule:

1. **Team members share codes here**
2. **Every person creates his/her own folder and put codes there.**
3. **Do not uploading training or testing datasets.**
4. **All the datasets must be moved to a folder with name "data"**
5. **All the executable files must be moved to a folder with name "apps”**

**[Discussion paper](https://paper.dropbox.com/doc/Ideas-are-posted-here-7w3BVIEZghSm5cpjjJoo8)**

**[Google Colab free GPU tutorial](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d)**


