# dwi_mining
This repository contains the code and resources developed for a data mining project applying the CRISP-DM process to diffusion-weighted imaging (DWI) data. The goal is to identify structural biomarkers that help distinguish neurotypical brains from those with neuropsychiatric disorders such as schizophrenia, bipolar disorder, and ADHD.

# Description
The UCLA Consortium for Neuropsychiatric Phenomics LA5c Study is a dataset whose main objective is to discover the physiological basis of variations in the phenotypes of psychological and neural systems, to advance the development of treatments and understanding of neuropsychiatric disorders.
The study included 272 participants, of whom 265 were preprocessed. These are divided into groups of healthy controls and patients diagnosed with ADHD, bipolar disorder, and schizophrenia. Metadata is available for each subject containing information relevant to this project: age, gender, and diagnosis.
The dataset includes various neuroimaging modalities, such as structural magnetic resonance imaging (MRI), diffusion-weighted imaging (DWI), and functional magnetic resonance imaging (fMRI), captured both at rest and during the performance of five cognitive tasks.
The data are organized according to the Brain Imaging Data Structure (BIDS) standard, and the images are preprocessed and standardized for use in research. This involves spatial normalization, filtering, noise and head motion correction, standardization to a reference template, and removal of facial features to protect identity.
The data to be used is in NIfTI (Neuroimaging Informatics Technology Initiative) format. In this format, the image matrices and metadata describing the image characteristics and acquisition parameters are stored together.

# Data Download
For downloading the data, change the drive path to a personal data folder, and run the notebook "data_download.ipynb" in Google Colab. This code will create the corresponding folders for a BIDS organization, following the standard in organizing neuroimaging data.
BIDS follows this format: 
<img width="1395" height="680" alt="image" src="https://github.com/user-attachments/assets/59e0a9b8-8851-4cd1-84c7-0c7a044c7eeb" />


# How to run
Explicar funcionamiento de los scripts y cómo correrlos

# Reference
Agregar referencia a la base de datos.

Authors: 
Aguilar Martinez Erick Yair, Chagüén Hernández Daniel Isidoro, Vera Garfias José Daniel 
