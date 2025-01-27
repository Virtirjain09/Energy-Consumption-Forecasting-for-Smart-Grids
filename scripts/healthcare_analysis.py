import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pydicom
import nibabel as nib
from sklearn.inspection import permutation_importance

# Data Pipeline functions (load_and_preprocess_data, process_ehr_data, etc.) remain unchanged
# Data Pipeline
def load_and_preprocess_data(data_sources):
    dfs = []
    for source, data in data_sources.items():
        if source == 'EHR':
            df = process_ehr_data(data)
        elif source == 'Clinical_Trials':
            df = process_clinical_trial_data(data)
        elif source == 'Demographics':
            df = process_demographic_data(data)
        elif source == 'Imaging':
            df = process_imaging_data(data)
        elif source == 'Public_Health':
            df = process_public_health_data(data)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, axis=1)
    
    # Preprocessing
    combined_df = pd.get_dummies(combined_df, columns=['Gender', 'Blood Type', 'Medical Condition'])
    
    # Split features and target
    X = combined_df.drop(columns=[col for col in combined_df.columns if 'Medical Condition_' in col])
    y = combined_df[[col for col in combined_df.columns if 'Medical Condition_' in col]].idxmax(axis=1)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(exclude=[np.number]).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return X, y, preprocessor, numeric_features, categorical_features

def process_ehr_data(data):
    # Process MIMIC-IV data
    return pd.DataFrame(data)

def process_clinical_trial_data(data):
    # Process data from ClinicalTrials.gov, EU Clinical Trials Register, etc.
    return pd.DataFrame(data)

def process_demographic_data(data):
    # Process data from Healthdata.gov, Data.gov, HMD
    return pd.DataFrame(data)

def process_imaging_data(data):
    # Process medical imaging data
    processed_data = []
    for image_path in data:
        if image_path.endswith('.dcm'):
            image = pydicom.dcmread(image_path)
            # Extract relevant features from DICOM image
            processed_data.append(extract_dicom_features(image))
        elif image_path.endswith('.nii') or image_path.endswith('.nii.gz'):
            image = nib.load(image_path)
            # Extract relevant features from NIfTI image
            processed_data.append(extract_nifti_features(image))
    return pd.DataFrame(processed_data)

def process_public_health_data(data):
    # Process data from WHO, Big Cities Health Inventory, HCUP
    return pd.DataFrame(data)

def extract_dicom_features(dicom_image):
    # Extract relevant features from DICOM image
    # This is a placeholder function, implement actual feature extraction
    return {'pixel_mean': np.mean(dicom_image.pixel_array)}

def extract_nifti_features(nifti_image):
    # Extract relevant features from NIfTI image
    # This is a placeholder function, implement actual feature extraction
    return {'voxel_mean': np.mean(nifti_image.get_fdata())}

# Experiment Design
def design_experiment(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    pipelines = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('model', SVC(kernel='rbf', random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    # Ensemble method (Voting Classifier)
    voting_clf = VotingClassifier(
        estimators=[(name, pipeline.named_steps['model']) for name, pipeline in pipelines.items()],
        voting='hard'
    )
    voting_clf.fit(preprocessor.fit_transform(X_train), y_train)
    y_pred_ensemble = voting_clf.predict(preprocessor.transform(X_test))
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    
    results['Ensemble'] = {
        'pipeline': voting_clf,
        'accuracy': ensemble_accuracy,
        'report': classification_report(y_test, y_pred_ensemble),
        'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble)
    }
    
    return results

def display_model_comparison(results):
    # Create accuracy comparison dataframe
    accuracy_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [result['accuracy'] for result in results.values()]
    })
    
    # Display accuracy table
    st.subheader("Model Accuracy Comparison")
    st.dataframe(accuracy_df.style.format({'Accuracy': '{:.2%}'}))
    
    # Plot accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=accuracy_df, x='Model', y='Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

def display_detailed_results(results):
    for name, result in results.items():
        with st.expander(f"{name} Detailed Results"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Classification Report")
                st.text(result['report'])
            
            with col2:
                st.write("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(result['confusion_matrix'], 
                          annot=True, 
                          fmt='d', 
                          cmap='Blues')
                plt.title(f'{name} Confusion Matrix')
                st.pyplot(fig)
                plt.close()

def display_feature_importance(results, X_test, y_test):
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['pipeline'].named_steps['model']
        try:
            perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
            
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False)
            
            st.subheader("Feature Importance Analysis")
            
            # Display importance table
            st.write("Top 10 Most Important Features")
            st.dataframe(feature_importance.head(10).style.format({'importance': '{:.4f}'}))
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not calculate feature importance: {str(e)}")

def analyze_clinical_outcomes(model, X, y):
    predictions = model.predict(X)
    outcomes = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions
    })
    
    st.write("Comparison of Actual vs Predicted Outcomes")
    st.dataframe(outcomes.head())
    
    fig, ax = plt.subplots()
    sns.scatterplot(x='Actual', y='Predicted', data=outcomes, ax=ax)
    plt.title('Actual vs Predicted Outcomes')
    plt.tight_layout()
    st.pyplot(fig)

def analyze_population_health(df):
    st.subheader("Population Health Analysis")
    
    age_groups = pd.cut(df['Age'], bins=[0, 18, 30, 50, 70, 100], labels=['0-18', '19-30', '31-50', '51-70', '71+'])
    
    demographics = df.groupby(['Gender', age_groups]).size().unstack()
    st.write("Demographic Breakdown")
    st.dataframe(demographics)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    demographics.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Population Demographics')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.legend(title='Age Group')
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")  # Use wide layout for better visualization
    st.title('Enhanced Healthcare Data Analysis Project')
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Clinical Outcomes", "Population Health"])
    
    with tab1:
        # Single file upload with memory optimization
        uploaded_file = st.file_uploader("Upload healthcare dataset (CSV)", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read CSV with optimized settings
                df = pd.read_csv(uploaded_file, dtype={
                    'Age': 'float32',
                    'Billing Amount': 'float32',
                    'Room Number': 'int32'
                })
                
                # Display data overview in an organized manner
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sample Data")
                    st.dataframe(df.head(), use_container_width=True)
                with col2:
                    st.subheader("Statistical Summary")
                    st.dataframe(df.describe(), use_container_width=True)
                
                # Target selection and feature limitation
                target_column = st.selectbox("Select target column for prediction", df.columns.tolist())
                
                # Limit features to prevent memory issues
                max_features = 5
                available_features = [col for col in df.columns if col != target_column]
                feature_columns = st.multiselect(
                    f"Select up to {max_features} most important features for prediction",
                    available_features,
                    default=available_features[:min(len(available_features), max_features)]
                )
                
                if len(feature_columns) > max_features:
                    st.warning(f"Please select no more than {max_features} features to avoid memory issues")
                    return
                
                if feature_columns:
                    # Prepare features and target with memory optimization
                    X = df[feature_columns].copy()
                    y = df[target_column].copy()
                    
                    # Sample data if too large
                    if len(X) > 10000:
                        sample_size = 10000
                        sample_idx = np.random.choice(len(X), sample_size, replace=False)
                        X = X.iloc[sample_idx]
                        y = y.iloc[sample_idx]
                    
                    # Preprocessing
                    numeric_features = X.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
                    categorical_features = X.select_dtypes(include=['object']).columns
                    
                    # Create optimized preprocessor
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', Pipeline([
                                ('imputer', SimpleImputer(strategy='mean')),
                                ('scaler', StandardScaler())
                            ]), numeric_features),
                            ('cat', Pipeline([
                                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
                            ]), categorical_features)
                        ]
                    )
                    
                    # Model training interface
                    st.subheader("Model Training Configuration")
                    col1, col2 = st.columns(2)
                    with col1:
                        test_size = st.slider("Test set size", 0.1, 0.5, 0.2)
                    with col2:
                        st.info(f"Training samples: {int(len(X) * (1-test_size))}\nTest samples: {int(len(X) * test_size)}")
                    
                    if st.button("Train and Evaluate Models"):
                        with st.spinner('Training models... This may take a few minutes.'):
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            results = train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
                            
                            # Display results in tabs
                            result_tab1, result_tab2, result_tab3 = st.tabs(["Model Comparison", "Detailed Results", "Feature Importance"])
                            
                            with result_tab1:
                                display_model_comparison(results)
                            
                            with result_tab2:
                                display_detailed_results(results)
                            
                            with result_tab3:
                                display_feature_importance(results, X_test, y_test)
                
                else:
                    st.error("Please select at least one feature for prediction")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try reducing the number of features or using a smaller dataset")
    
    with tab2:
        st.subheader("Clinical Outcomes Analysis")
        if 'results' in locals() and 'X_test' in locals() and 'y_test' in locals():
            best_model = max(results, key=lambda x: results[x]['accuracy'])
            analyze_clinical_outcomes(results[best_model]['pipeline'], X_test, y_test)
        else:
            st.info("Train models first to see clinical outcomes analysis")

    with tab3:
        st.subheader("Population Health Analysis")
        if 'df' in locals():
            analyze_population_health(df)
        else:
            st.info("Upload data to see population health analysis")

if __name__ == "__main__":
    main()
