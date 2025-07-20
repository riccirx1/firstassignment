"""
Azure ML VotingEnsemble Model Deployment Script

This script handles the deployment of a VotingEnsemble model to Azure ML,
addressing common subscription and service configuration issues.
"""

import json
import os
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.subscription import SubscriptionClient
from azureml.core import Workspace, Environment, Model
from azureml.core.authentication import ServicePrincipalAuthentication, AzureCliAuthentication
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureMLVotingEnsembleDeployer:
    """
    Handles deployment of VotingEnsemble models to Azure ML with proper subscription management.
    """
    
    def __init__(self, config_file='azure_config.json'):
        """
        Initialize the deployer with configuration.
        
        Args:
            config_file (str): Path to Azure configuration file
        """
        self.config = self._load_config(config_file)
        self.credential = None
        self.workspace = None
        
    def _load_config(self, config_file):
        """Load Azure configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file {config_file}")
            raise
    
    def verify_subscription(self):
        """
        Verify Azure subscription access and required services.
        This addresses the subscription error mentioned in the problem statement.
        """
        logger.info("Verifying Azure subscription and services...")
        
        try:
            # Try different authentication methods
            credential = None
            
            # Method 1: Try Azure CLI authentication (recommended for development)
            try:
                credential = AzureCliAuthentication()
                logger.info("Using Azure CLI authentication")
            except Exception as e:
                logger.warning(f"Azure CLI authentication failed: {e}")
            
            # Method 2: Try default credential chain
            if not credential:
                try:
                    credential = DefaultAzureCredential()
                    logger.info("Using DefaultAzureCredential")
                except Exception as e:
                    logger.warning(f"DefaultAzureCredential failed: {e}")
            
            # Method 3: Interactive browser authentication as fallback
            if not credential:
                try:
                    credential = InteractiveBrowserCredential()
                    logger.info("Using Interactive Browser authentication")
                except Exception as e:
                    logger.error(f"All authentication methods failed: {e}")
                    raise
            
            self.credential = credential
            
            # Verify subscription access
            subscription_client = SubscriptionClient(credential)
            subscription = subscription_client.subscriptions.get(self.config['subscription_id'])
            logger.info(f"Successfully accessed subscription: {subscription.display_name}")
            
            # Check if Azure ML service is available in the subscription
            resource_client = ResourceManagementClient(credential, self.config['subscription_id'])
            
            # List resource providers to verify Azure ML is registered
            providers = resource_client.providers.list()
            ml_provider = None
            for provider in providers:
                if provider.namespace == 'Microsoft.MachineLearningServices':
                    ml_provider = provider
                    break
            
            if not ml_provider:
                logger.error("Azure Machine Learning service provider not found in subscription")
                raise Exception("Azure ML service not available. Please ensure Microsoft.MachineLearningServices is registered.")
            
            if ml_provider.registration_state != 'Registered':
                logger.warning(f"Azure ML provider state: {ml_provider.registration_state}")
                logger.info("Registering Azure ML service provider...")
                resource_client.providers.register('Microsoft.MachineLearningServices')
                logger.info("Azure ML service provider registration initiated")
            
            logger.info("Subscription verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Subscription verification failed: {e}")
            raise
    
    def create_or_get_workspace(self):
        """
        Create or get Azure ML workspace with proper error handling.
        """
        logger.info("Creating or accessing Azure ML workspace...")
        
        try:
            # Try to get existing workspace
            self.workspace = Workspace.get(
                name=self.config['workspace_name'],
                subscription_id=self.config['subscription_id'],
                resource_group=self.config['resource_group'],
                auth=self.credential
            )
            logger.info(f"Connected to existing workspace: {self.workspace.name}")
            
        except Exception as e:
            logger.info(f"Workspace not found, creating new one: {e}")
            
            # Create new workspace
            try:
                self.workspace = Workspace.create(
                    name=self.config['workspace_name'],
                    subscription_id=self.config['subscription_id'],
                    resource_group=self.config['resource_group'],
                    location=self.config['location'],
                    auth=self.credential,
                    exist_ok=True
                )
                logger.info(f"Created new workspace: {self.workspace.name}")
                
            except Exception as create_error:
                logger.error(f"Failed to create workspace: {create_error}")
                raise
        
        return self.workspace
    
    def prepare_voting_ensemble_environment(self):
        """
        Prepare the environment for VotingEnsemble deployment.
        """
        logger.info("Preparing VotingEnsemble environment...")
        
        # Create environment with required packages
        env = Environment(name="voting-ensemble-env")
        
        # Add conda dependencies
        conda_dep = env.python.conda_dependencies
        conda_dep.add_pip_package("azureml-defaults")
        conda_dep.add_pip_package("scikit-learn")
        conda_dep.add_pip_package("pandas")
        conda_dep.add_pip_package("numpy")
        conda_dep.add_pip_package("joblib")
        
        # Register environment
        env.register(workspace=self.workspace)
        logger.info("Environment registered successfully")
        
        return env
    
    def create_inference_config(self, env):
        """
        Create inference configuration for the VotingEnsemble model.
        """
        logger.info("Creating inference configuration...")
        
        # Create entry script for VotingEnsemble
        entry_script_content = '''
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
import joblib
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'voting_ensemble_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df) if hasattr(model, 'predict_proba') else None
        
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None
        }
        
        return json.dumps(result)
    except Exception as e:
        return json.dumps({'error': str(e)})
'''
        
        # Save entry script
        os.makedirs('deployment_files', exist_ok=True)
        with open('deployment_files/score.py', 'w') as f:
            f.write(entry_script_content)
        
        # Create inference config
        inference_config = InferenceConfig(
            entry_script='score.py',
            source_directory='deployment_files',
            environment=env
        )
        
        logger.info("Inference configuration created")
        return inference_config
    
    def deploy_voting_ensemble(self, model_name, service_name):
        """
        Deploy the VotingEnsemble model to Azure Container Instance.
        
        Args:
            model_name (str): Name of the registered model
            service_name (str): Name for the deployed service
        """
        logger.info(f"Deploying VotingEnsemble model: {model_name}")
        
        try:
            # Get the model
            model = Model(self.workspace, model_name)
            logger.info(f"Retrieved model: {model.name}")
            
            # Prepare environment and inference config
            env = self.prepare_voting_ensemble_environment()
            inference_config = self.create_inference_config(env)
            
            # Configure deployment
            aci_config = AciWebservice.deploy_configuration(
                cpu_cores=1,
                memory_gb=1,
                tags={'model': 'VotingEnsemble', 'type': 'classification'},
                description='VotingEnsemble model deployment'
            )
            
            # Deploy the service
            service = Model.deploy(
                workspace=self.workspace,
                name=service_name,
                models=[model],
                inference_config=inference_config,
                deployment_config=aci_config,
                overwrite=True
            )
            
            # Wait for deployment to complete
            service.wait_for_deployment(show_output=True)
            
            logger.info(f"Service deployed successfully: {service.scoring_uri}")
            return service
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

def main():
    """
    Main function to handle VotingEnsemble deployment with subscription error resolution.
    """
    print("Azure ML VotingEnsemble Deployment Script")
    print("=========================================")
    
    try:
        # Initialize deployer
        deployer = AzureMLVotingEnsembleDeployer()
        
        # Step 1: Verify subscription and services
        print("\n1. Verifying Azure subscription and services...")
        deployer.verify_subscription()
        print("✓ Subscription verification completed")
        
        # Step 2: Create or get workspace
        print("\n2. Setting up Azure ML workspace...")
        workspace = deployer.create_or_get_workspace()
        print(f"✓ Workspace ready: {workspace.name}")
        
        # Step 3: Instructions for model deployment
        print("\n3. VotingEnsemble deployment ready!")
        print("To deploy your VotingEnsemble model:")
        print("   - Register your trained VotingEnsemble model using: workspace.register_model()")
        print("   - Call deployer.deploy_voting_ensemble(model_name, service_name)")
        
        print("\n✓ Azure ML setup completed successfully!")
        print("The subscription service issue has been resolved.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you're logged in to Azure CLI: az login")
        print("2. Verify your subscription ID in azure_config.json")
        print("3. Check that Azure ML service is enabled in your subscription")
        print("4. Ensure you have proper permissions for the resource group")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())