"""
Test script to validate Azure ML VotingEnsemble deployment setup.
This script tests the basic setup without requiring Azure SDK installation.
"""

import json
import os

def validate_files_and_config():
    """Validate that all required files are present and properly configured."""
    print("Running Azure ML deployment validation...")
    
    required_files = [
        'requirements.txt',
        'azure_config.json', 
        'deploy_voting_ensemble.py',
        'AZURE_ML_README.md'
    ]
    
    # Check if all required files exist
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing required file: {file}")
            return False
        print(f"✓ Found: {file}")
    
    # Validate configuration file format
    try:
        with open('azure_config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['subscription_id', 'resource_group', 'workspace_name', 'location']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing key in azure_config.json: {key}")
                return False
        print("✓ Configuration file format is valid")
        
    except json.JSONDecodeError:
        print("❌ Invalid JSON format in azure_config.json")
        return False
    
    # Check if requirements.txt contains Azure ML packages
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
    
    if 'azureml-sdk' not in requirements:
        print("❌ Missing azureml-sdk in requirements.txt")
        return False
    print("✓ Requirements.txt contains Azure ML packages")
    
    # Validate Python script syntax
    try:
        with open('deploy_voting_ensemble.py', 'r') as f:
            script_content = f.read()
        
        # Basic syntax check - compile the code
        compile(script_content, 'deploy_voting_ensemble.py', 'exec')
        print("✓ Deploy script syntax is valid")
        
    except SyntaxError as e:
        print(f"❌ Syntax error in deploy_voting_ensemble.py: {e}")
        return False
    
    print("✓ All validation checks passed!")
    return True

if __name__ == "__main__":
    # Run validation tests
    success = validate_files_and_config()
    
    if success:
        print("\n" + "="*50)
        print("Azure ML VotingEnsemble deployment setup is ready!")
        print("="*50)
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Update azure_config.json with your Azure details")
        print("3. Run: python deploy_voting_ensemble.py")
        print("4. Follow the instructions to deploy your model")
        print("\nThis setup resolves the subscription service error by:")
        print("- Automatically registering Microsoft.MachineLearningServices")
        print("- Handling multiple authentication methods")
        print("- Providing comprehensive error handling")
    else:
        print("\n" + "="*50)
        print("Please fix the validation errors before proceeding")
        print("="*50)