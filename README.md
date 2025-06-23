# Network Security
This project is a FastAPI-based web application designed for network security tasks, with a focus on training machine learning models and making real-time predictions on network traffic data, such as detecting phishing attempts. The system features an end-to-end CI/CD pipeline for streamlined development and deployment. It leverages Docker for containerization and integrates with AWS services, including Amazon ECR for image hosting, S3 for secure data storage, and EC2 for scalable application deployment. MongoDB is used for storing training data, model metadata, and prediction logs. The project architecture ensures modularity, scalability, and automation across the machine learning lifecycle—from data ingestion and model training to inference and monitoring.

## Project Structure
```
.
├── .env                         # Environment variables configuration
├── .gitignore                   # Specifies files and folders to be ignored by Git
├── app.py                       # FastAPI application instance
├── main.py                      # Main script to launch the FastAPI server
├── push_data.py                 # Script to upload data to MongoDB
├── Dockerfile                   # Dockerfile for containerizing the application
├── README.md                    # Project documentation and usage guide
├── requirements.txt             # List of required Python packages
├── setup.py                     # Setup script for packaging the application
├── test_mongodb.py              # Script to test MongoDB connectivity
├── .github/
│   └── workflows/
│       └── main.yml             # GitHub Actions workflow for CI/CD
├── data_schema/
│   └── schema.yaml              # YAML file specifying data schema
├── final_models/                # Serialized ML model and preprocessor
│   ├── model.pkl                # Trained ML model
│   └── preprocessor.pkl         # Preprocessing pipeline
├── network_data/
│   └── phisingData.csv          # Example phishing dataset
├── network_security/            # Core application package
│   ├── __init__.py              # Package initializer
│   ├── cloud/                   # Cloud-related utilities (e.g., S3 sync)
│   │   ├── __init__.py
│   │   └── s3_syncer.py
│   ├── components/              # Pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── constant/                # Global constants and configs
│   │   ├── __init__.py
│   │   └── training_pipeline/
│   │       ├── __init__.py
│   ├── entity/                  # Entity/data schema definitions
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/               # Custom exception classes
│   │   ├── __init__.py
│   │   └── exception.py
│   ├── logging/                 # Logging configuration
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── pipeline/                # Pipeline orchestrators
│   │   ├── __init__.py
│   │   ├── batch_prediction.py
│   │   └── training_pipeline.py
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── main_utils/          # General-purpose helpers
│       └── ml_utils/            # ML-specific helpers
├── prediction_output/           # Directory for model predictions
│   └── output.csv               # Output from predictions
└── templates/                   # HTML templates for rendering FastAPI views
```
## Set Up Instruction
### 1. Clone the repository
```
git clone https://github.com/<your-username>/NetworkSecurity.git
cd NetworkSecurity
```
### 2. Create a Virtual Environment & Install dependencies
```
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```
### 3. Set Up Environment Variables in .env file and in Github Secret
```
MONGO_DB_URL = your_mongo_db_url
AWS_ACCESS_KEY_ID = your_acess_key_id
AWS_SECRET_ACCESS_KEY = your_access_key
AWS_REGION = your_region
ECR_REPOSITORY_NAME = your_ecr_repo_name
AWS_ECR_LOGIN_URI = your_ecr_uri
```
### 4. Run the application in your local machine
```
python app.py
```
You access the API documentation at http://127.0.0.1:8000/docs.
### Optional: Run the application in AWS S3 EC2 Instance
- Step 1: Create an EC2 Instance
- Step 2: Connect to the EC2 Instance & Set Up Docker
Run the following command:
```
# Optional: Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Required: Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add Docker permissions to user
sudo usermod -aG docker ubuntu
newgrp docker
```
- Step 3: Set Up GitHub Actions Self-Hosted Runner
  - Go to your GitHub repo → Settings → Actions → Runners
  - Click "New self-hosted runner"
    - Choose:
    - OS: Linux
    - Architecture: x64
  - Copy and run the commands on your EC2
- Step 4:  Push Code to GitHub and Open the App in EC2 instance
## CI/CD Pipeline
The project uses GitHub Actions for CI/CD, defined in `.github/workflows/main.yml`.  is structured into three main stages:
### Continuous Integration (CI)
- Checkout code: Pulls the latest code from the GitHub repository using the actions/checkout action.
- Lint the repository: Runs a placeholder linting command to enforce code quality standards (replace with tools like flake8, black, etc.).
- Run unit tests: Executes test commands to validate application logic and catch errors early in the pipeline.
### Continuous Delivery (CD)
- Checkout code: Pulls the source code again for the delivery stage.
- Install utilities: Installs required tools (jq, unzip, etc.) on the runner.
- Configure AWS credentials: Uses GitHub Secrets to authenticate with AWS and allow access to ECR.
- Login to Amazon ECR: Authenticates the Docker client with AWS ECR using the amazon-ecr-login GitHub Action.
- Build and push Docker image:
  - Builds a Docker image of the FastAPI application.
  - Tags the image as latest.
  - Pushes it to the Amazon ECR repository.
### Continuous Deployment (CD)
- Run on self-hosted EC2 runner: Uses your EC2 instance (with a self-hosted GitHub Actions runner) to handle deployment.
- Configure AWS credentials: Re-authenticates on the EC2 runner to pull from ECR securely.
- Login to Amazon ECR: Logs in again to ensure Docker can pull from your private ECR repository.
- Pull latest image from ECR: Downloads the most recent Docker image from ECR.
- Run Docker container: Runs the application in a container
  - Sets AWS credentials as environment variables (if needed inside the app).
  - Maps port 8080 for public access.
- Clean previous images and containers: Cleans up unused Docker containers, images, and other resources to free disk space.
## Final Result
![image]<img width="1115" alt="Screenshot 2025-06-22 at 22 34 12" src="https://github.com/user-attachments/assets/8ae4b638-ee78-4da1-8ab2-96c573a14301" />


