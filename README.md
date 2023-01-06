<h1>COVID-19 2021 Kaggle Competition 17th Place Solution</h1>

This repository contains the code for the classifier part of the solution.

It's in <code>src/classifier</code>.

The configs for the experiment are in src/config.

To run an experiment, you would first have to download the dataset from Kaggle, then follow these steps:

<ol>
  <li>Convert data from DICOM to png using <code>src/data_prep/convert_dicom_to_png.py</code>.</li>
  <li>Create opacity masks from lung labels using competition's train_image_level.csv and <code>src/data_prep/create_opacity_masks.py</code>.</li>
  <li>Prepare image level dataframe using <code>src/data_prep/prepare_image_level_df.py</code>.</li>
  <li>Create stratified k folds using <code>src/data_prep/create_folds.py</code>.</li>
  <li>Create a config as per the configs in <code>src/classifier/config</code>.</li>
  <li>Run a training job as per <code>src/classifier/train.sh</code>.</li>  
</ol>
