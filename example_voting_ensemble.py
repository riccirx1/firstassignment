"""
Example script showing how to create and deploy a VotingEnsemble model in Azure ML.
This demonstrates the complete workflow from model creation to deployment.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import joblib
import os

def create_sample_voting_ensemble():
    """
    Create a sample VotingEnsemble model for demonstration.
    In practice, you would load your own data and train your specific model.
    """
    print("Creating sample VotingEnsemble model...")
    
    # Create sample dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create individual models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    svm_model = SVC(probability=True, random_state=42)
    
    # Create VotingEnsemble
    voting_ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr', lr_model),
            ('svm', svm_model)
        ],
        voting='soft'  # Use soft voting for probability predictions
    )
    
    # Train the ensemble
    print("Training VotingEnsemble...")
    voting_ensemble.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = voting_ensemble.score(X_train, y_train)
    test_score = voting_ensemble.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    return voting_ensemble, X_test[:5], y_test[:5]  # Return model and sample test data

def save_model_for_azure(model, model_name="voting_ensemble_model"):
    """Save the model in a format suitable for Azure ML deployment."""
    
    # Create model directory
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path

def register_model_in_azure(model_path, workspace):
    """
    Register the model in Azure ML workspace.
    This function would be called after setting up the Azure ML workspace.
    """
    from azureml.core import Model
    
    try:
        model = Model.register(
            workspace=workspace,
            model_path=model_path,
            model_name="voting-ensemble-model",
            description="VotingEnsemble model with RF, LR, and SVM",
            tags={"type": "classification", "algorithm": "VotingEnsemble"}
        )
        print(f"Model registered successfully: {model.name}, Version: {model.version}")
        return model
    except Exception as e:
        print(f"Error registering model: {e}")
        raise

def demonstrate_deployment_workflow():
    """
    Demonstrate the complete workflow for VotingEnsemble deployment.
    """
    print("VotingEnsemble Azure Deployment Workflow Demo")
    print("=" * 50)
    
    # Step 1: Create and train the VotingEnsemble model
    model, sample_X, sample_y = create_sample_voting_ensemble()
    
    # Step 2: Save the model
    model_path = save_model_for_azure(model)
    
    # Step 3: Test local predictions
    print("\nTesting local predictions...")
    predictions = model.predict(sample_X)
    probabilities = model.predict_proba(sample_X)
    
    print("Sample predictions:")
    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, sample_y)):
        print(f"  Sample {i+1}: Predicted={pred}, Actual={actual}, Confidence={max(prob):.3f}")
    
    # Step 4: Deployment instructions
    print("\n" + "=" * 50)
    print("Next Steps for Azure Deployment:")
    print("=" * 50)
    print("1. Ensure Azure ML workspace is set up:")
    print("   python deploy_voting_ensemble.py")
    print()
    print("2. Register the model in Azure ML workspace:")
    print("   # Add this to your deployment script:")
    print("   model = Model.register(")
    print("       workspace=workspace,")
    print(f"       model_path='{model_path}',")
    print("       model_name='voting-ensemble-model'")
    print("   )")
    print()
    print("3. Deploy the model:")
    print("   deployer.deploy_voting_ensemble('voting-ensemble-model', 'voting-service')")
    print()
    print("4. Test the deployed service with sample data")
    
    # Create sample test data file
    test_data = {
        "data": sample_X.tolist()
    }
    
    import json
    with open("sample_test_data.json", "w") as f:
        json.dump(test_data, f)
    
    print(f"\nSample test data saved to: sample_test_data.json")
    print("Use this data to test your deployed service")

if __name__ == "__main__":
    demonstrate_deployment_workflow()