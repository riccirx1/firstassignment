# firstassignment
CSS Course First Assignment

## Original Content
This repository contains a CSS course assignment featuring a website about springtime walks in Michigan.

## Azure ML VotingEnsemble Deployment (New)

**Problem Solved:** Resolves the Azure subscription service error when deploying VotingEnsemble models.

### Quick Start for Azure ML

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Azure settings:**
   Edit `azure_config.json` with your Azure details

3. **Deploy VotingEnsemble model:**
   ```bash
   python deploy_voting_ensemble.py
   ```

4. **Run example workflow:**
   ```bash
   python example_voting_ensemble.py
   ```

### Files Added for Azure ML
- `deploy_voting_ensemble.py` - Main deployment script with subscription fix
- `azure_config.json` - Azure configuration template
- `requirements.txt` - Azure ML SDK dependencies
- `example_voting_ensemble.py` - Example model creation and deployment
- `test_deployment.py` - Validation tests
- `AZURE_ML_README.md` - Detailed Azure ML documentation

The Azure ML implementation automatically handles the subscription service registration that was causing deployment errors.
