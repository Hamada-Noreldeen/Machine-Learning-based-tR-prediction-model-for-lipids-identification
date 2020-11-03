# ML-based-tR-prediction-model-for-lipids-identification
Machine learning (ML)-based retention time (tR) prediction model for lipids identification 
In this study, a retention time prediction model was developed based on molecular descriptors and molecular fingerprints by using a machine learning method to help increase confidence in lipid annotation and reduce the identification errors. Furthermore, the comparison between molecular descriptors and molecular fingerprints was performed. Then, datasets were applied to validate the applicability and stability of the retention time predictions model. For possible retention time deviations between different instruments or acquisition batches, a linear retention time calibration method was used. Also, for the retention time difference between different chromatographic systems (CS), a linear relationship was constructed to transfer the retention time from the old CS to a new CS with the help of the ML model.
The steps are as follows: 
1-	The data has been randomly splatted into (2:1), then we used selected molecular descriptors for model training (Folder1).
2-	In the model, we used a general code to define our suitable parameters for the random forest algorithm by using Bayesian for getting the best parameter. Also, cross-validation (CV) at K=10, has been considered (Folder1).
3-	After defining our final model, we applied it by using molecular fingerprints (MF), as well as, by using both molecular fingerprints + molecular descriptors (MD_MF), (Folder1).
4-	Also, we used two datasets as validation sets to prove our findings (Folder2 and 3).
5-	Due to the better result that we got from the MD-based model, so we used it for transferring from one chromatographic system (CS) into another - two CSs were different in gradients, matrix, and running time, with the same buffers and columns - (Folder4).
6-	Of note, in Folder5, we didn’t use our model, we just made tR celebration from one CS (Main dataset) into another (new CS)- the two CSs were the same in buffers and columns, but different in acquisition batches, matrix, or instruments -.

