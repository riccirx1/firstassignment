# Azure ML VotingEnsemble Deployment

This repository now includes Azure Machine Learning infrastructure to deploy VotingEnsemble models and resolve subscription service issues.

## Problem Addressed

The common error "there is not a subscription for the service" when deploying VotingEnsemble models in Azure ML is typically caused by:

1. **Missing Azure ML Service Registration**: The Microsoft.MachineLearningServices provider is not registered in your subscription
2. **Authentication Issues**: Improper authentication setup
3. **Workspace Configuration**: Missing or incorrect workspace configuration

## Solution

This implementation provides a comprehensive solution that:

### 1. Verifies and Registers Required Services

The script automatically:
- Checks if Azure ML service is registered in your subscription
- Registers `Microsoft.MachineLearningServices` provider if needed
- Verifies authentication and subscription access

### 2. Handles Multiple Authentication Methods

Supports various authentication approaches:
- Azure CLI authentication (recommended for development)
- DefaultAzureCredential (for production environments)
- Interactive Browser authentication (fallback)

### 3. Provides Robust Error Handling

Includes comprehensive error handling and troubleshooting guidance.

## Setup Instructions

### Prerequisites

1. **Azure Subscription**: Ensure you have an active Azure subscription
2. **Azure CLI**: Install and login using `az login`
3. **Python Environment**: Python 3.7+ with pip

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Configure Azure settings in `azure_config.json`:
```json
{
    "subscription_id": "your-actual-subscription-id",
    "resource_group": "your-resource-group-name", 
    "workspace_name": "your-workspace-name",
    "location": "eastus"
}
```

### Usage

#### Basic Deployment Setup

```python
from deploy_voting_ensemble import AzureMLVotingEnsembleDeployer

# Initialize deployer
deployer = AzureMLVotingEnsembleDeployer()

# Verify subscription and setup workspace
deployer.verify_subscription()
workspace = deployer.create_or_get_workspace()

# Deploy your VotingEnsemble model
# (after training and registering your model)
service = deployer.deploy_voting_ensemble("your-model-name", "voting-ensemble-service")
```

#### Command Line Usage

```bash
python deploy_voting_ensemble.py
```

## Service Subscription Resolution

The script addresses the subscription error by:

1. **Automatic Service Registration**: 
   - Checks if `Microsoft.MachineLearningServices` is registered
   - Automatically registers the service if missing
   
2. **Authentication Verification**:
   - Tests multiple authentication methods
   - Provides clear error messages for authentication issues
   
3. **Subscription Validation**:
   - Verifies access to the specified subscription
   - Checks resource provider availability

## Troubleshooting

If you encounter subscription errors:

1. **Login to Azure CLI**:
   ```bash
   az login
   ```

2. **Verify Subscription**:
   ```bash
   az account show
   az account set --subscription "your-subscription-id"
   ```

3. **Check Resource Providers**:
   ```bash
   az provider show --namespace Microsoft.MachineLearningServices
   ```

4. **Register ML Service** (if needed):
   ```bash
   az provider register --namespace Microsoft.MachineLearningServices
   ```

## Files Added

- `requirements.txt`: Azure ML SDK dependencies
- `azure_config.json`: Azure configuration template
- `deploy_voting_ensemble.py`: Main deployment script with subscription error resolution
- `README.md`: This documentation (updated)

## Next Steps

1. Update `azure_config.json` with your actual Azure details
2. Train your VotingEnsemble model
3. Register the model in Azure ML workspace
4. Deploy using the provided script

The subscription service error should now be resolved with proper service registration and authentication handling.