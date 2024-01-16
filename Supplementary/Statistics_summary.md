# Overall Training/Test summary

## Usage of Upsamling to balance dataset:
- total dataset after upsampling: 680 
- training dataset: 544 
- test dataset: 136

## Usage of <b> nested stratified cross validation </b> for hyperparameter tuning.
Nested CV Recall: 0.8735 (0.7846-0.9412)

# Grid parameter search optimized for: <b> Recall </b>
Best Parameters from Nested CV: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 45}

# Relevance of features in training:
## Feature Importances:

### cost function derived on relative transition propensities for each AA to AA mutation
cost: 0.2461494416979599
### this is a combinatorial feature that takes into account change in hydrophobicity weight and number of atoms upon mutation.
hydro_weight_num_abs_dhydro: 0.10218954533360865
### a conservation score but derived WITHIN FLN IG domains (instead of blast against FLNC related proteins). Very strong feature and we already observed this similar as described in our paper : where we showed conservation among the IG domains.
lockless_cons: 0.20164014223918467
### korpm ddg predictor (but absolute values because apparently the benign mutations sit around ddg 0 and pathological mutations populate either more extreme stable or destabilzed values)
ddg_abs: 0.19042543674313828
### difference between WT and MUT aa in terms of clashes. benign mutations tend to introduce less clashes.
clashcounter_diff: 0.0805225184881575
### Chemical surroundings and solvent accessibility. 10A volume scanned.
SAP_scores: 0.17907291549795096

## The confusion matrix based on test datast: 136 mutations.
Confusion Matrix:
[[62  0]
 [15 59]]

# Test Set Metrics with 95% CI based on 1000 boostrap sample runs:

### each sample run drew different test sets and made predictions based on the best found hyperparameters in step 1 (nested CV)

# Test Set Metrics (95% confidence interv.):

## Accuracy: 0.8894 ((0.8382352941176471, 0.9338235294117647))
## Precision: 1.0000 ((1.0, 1.0))
## Recall: 0.7972 ((0.7066282051282051, 0.8831404103143233))
## F1 Score: 0.8864 ((0.8280985667293234, 0.937944297082228))
## AUC-ROC: 0.8986 ((0.8533141025641026, 0.9415702051571616))
## F2 Score: 0.8305 (0.7507-0.9043)
## Matthews Correlation Coefficient (MCC): 0.8006 (0.7189-0.8759)