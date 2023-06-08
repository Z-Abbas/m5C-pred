# m5C-pred
XGBoost Framework with Feature Selection for the Prediction of RNA N5-methylcytosine sites

## Abstract
5-methylcytosine (m5C) is indeed a critical post-transcriptional alteration that is widely present in various kinds of RNAs and is crucial to the fundamental biological processes. By correctly identifying the m5C-methylation sites on RNA, clinicians can more clearly comprehend the precise function of these m5C-sites in different biological processes. Due to their effectiveness and affordability, computational methods have received greater attention over the last few years for the identification of methylation sites in various species. To precisely identify RNA m5C locations in five different species including Homo sapiens, Arabidopsis thaliana, Mus musculus, Drosophila melanogaster, and Danio rerio, we proposed a more effective and accurate model named m5C-pred. To create m5C-pred, five distinct feature encoding techniques were combined to extract features from the RNA sequence and then used SHAP (SHapley Additive exPlanations) to choose the best features among them, followed by XGBoost as a classifier. We applied the novel optimization method called OPTUNA to quickly and efficiently determine the best hyperparameters. Finally, the proposed model was evaluated using independent test datasets and compared the results with the previous methods. Our approach, m5C- pred, is anticipated to be useful for accurately identifying m5C sites, outperforming the currently available state- of-the-art techniques.

![block_rev](https://user-images.githubusercontent.com/80881943/210732228-7d68b0ce-eac7-4cbd-ad47-1746b1d8f876.jpg)

**For more details, please refer to the [paper](https://www.cell.com/molecular-therapy-family/molecular-therapy/fulltext/S1525-0016(23)00272-1)**

## Help
To test the model, we have provided all the weight files along with other required files. Just the main Python file has to be executed.

To execute the program, please adhere to the procedures below.

- Download the complete repository to your PC.  
- Now, run the main.py file using command line as:
    - python main.py [data_path] [specie]
- It will provide the prediction in results.txt file in the same working directory as:
    - Sequence ID
    - Location
    - Sequence Background
    - Probability
    - Result  
   
## Specifications
- Python 3.7
- tensorflow 2.4.1
- keras 2.4.3
- numpy 1.18.5
- pandas 1.2.4
