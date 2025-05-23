## Installation
1. Install all necessary packages using:
   ```
   pip install -r requirements.txt
   ```

## Usage of Logistic Regression CCD Model
1. Define the model with desired parameters:
   ```python
   model = LogRegCCD(
       lambdas=np.logspace(-4, 0, 20),
       max_iter=1000,
       tol=1e-6
   )
   ```

2. Train, validate and predict with the model:
   ```python
   # Train the model
   model.fit(X_train, y_train)
   
   # Validate model performance
   model.validate(X_valid, y_valid, measure)
   
   # Make probability predictions
   predictions = model.predict_proba(X_test)
   ```
   
   For a complete example, see `LogRegCCD_example_of_use.ipynb`.

## Data and Evaluation
1. All benchmark datasets for comparison are available in the `data` folder, along with functions for model generation.

2. The file `compare_real_datasets.py` contains methods to generate evaluation metrics and coefficients for comparison and save them as .txt files, to run it:
   `python compare_real_datasets.py --data path/to/data --output path/to/output_folder`

4. The file `LogRegCCD_LogRegSklearn_Breast_Ionosphere.ipnyb` shows sample printing results in our notebook format for two datasets and shows the results (one of us was working in a notebook, but the script `compare_real_datasets.py` can be used to compare all sets).

5. The notebook `test_synthetic.ipynb` performs similar evaluation on synthetic data.

6. All model coefficients are saved in the `data/coefficients` folder.
