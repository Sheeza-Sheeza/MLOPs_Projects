# Vehicle Insurance Prediction — MLOps Project

End-to-end MLOps pipeline for predicting vehicle insurance customer response. Data is ingested from **MongoDB Atlas**, models are trained and versioned with **AWS S3**, and predictions are served through a **FastAPI** web app deployed to **EC2** via **GitHub Actions** and **Docker**.

## Features

- **Modular ML pipeline** — data ingestion, validation, transformation, training, evaluation, and model pushing
- **MongoDB Atlas** — source-of-truth dataset storage with batched export for large collections
- **AWS S3 model registry** — production models pulled at inference time
- **FastAPI web UI** — form-based vehicle insurance prediction
- **CI/CD** — GitHub Actions builds a Docker image, pushes to ECR, and deploys to a self-hosted EC2 runner

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| ML | scikit-learn, imbalanced-learn, pandas, numpy |
| API | FastAPI, Uvicorn, Jinja2 |
| Data | MongoDB Atlas (pymongo), AWS S3 (boto3) |
| DevOps | Docker, Amazon ECR, EC2, GitHub Actions |

## ML Pipeline

```
MongoDB Atlas
      │
      ▼
Data Ingestion ──► Data Validation ──► Data Transformation
                                              │
                                              ▼
                                        Model Trainer
                                              │
                              ┌───────────────┴───────────────┐
                              ▼                               ▼
                      Model Evaluation                  Model Pusher
                              │                               │
                              └───────────► AWS S3 ◄────────────┘
                                              │
                                              ▼
                                    Prediction (FastAPI)
```

### Pipeline components

| Component | Description |
|-----------|-------------|
| **Data Ingestion** | Fetches records from MongoDB and splits into train/test CSVs |
| **Data Validation** | Validates schema, dtypes, and drift against `config/schema.yaml` |
| **Data Transformation** | Feature engineering, scaling, and preprocessing object creation |
| **Model Trainer** | Trains a Random Forest classifier (config in `config/model.yaml`) |
| **Model Evaluation** | Compares new model against the production model in S3 |
| **Model Pusher** | Pushes accepted models to the S3 model registry |

## Project Structure

```
MLOPs_Projects/
├── app.py                      # FastAPI application entry point
├── config/
│   ├── schema.yaml             # Dataset schema and feature definitions
│   └── model.yaml              # Model hyperparameters
├── src/
│   ├── components/             # Pipeline stage implementations
│   ├── configuration/          # MongoDB and AWS connection setup
│   ├── constants/              # Project-wide constants
│   ├── data_access/            # MongoDB data access layer
│   ├── entity/                 # Config and artifact entity classes
│   ├── pipline/                # Training and prediction pipelines
│   ├── cloud_storage/          # AWS S3 utilities
│   ├── logger/                 # Logging configuration
│   └── utils/                  # Shared utilities
├── static/                     # CSS and static assets
├── template/                   # Jinja2 HTML templates
├── .github/workflows/aws.yaml  # CI/CD workflow
├── Dockerfile
└── requirements.txt
```

## Prerequisites

- Python 3.12+
- MongoDB Atlas cluster with vehicle insurance data
- AWS account with S3 bucket and ECR repository
- (For deployment) EC2 instance with Docker and a GitHub self-hosted runner

## Local Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/Sheeza-Sheeza/MLOPs_Projects.git
cd MLOPs_Projects

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
MONGODB_URL=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

Alternatively, set them in your shell:

```powershell
# PowerShell
$env:MONGODB_URL = "mongodb+srv://..."
$env:AWS_ACCESS_KEY_ID = "..."
$env:AWS_SECRET_ACCESS_KEY = "..."
```

```bash
# Bash
export MONGODB_URL="mongodb+srv://..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

### 3. Run the application locally

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 4. Run the training pipeline

Trigger training via the API:

```
GET http://127.0.0.1:5000/train
```

Or run the pipeline directly:

```bash
python -c "from src.pipline.training_pipeline import TrainPipeline; TrainPipeline().run_pipeline()"
```

## Docker

Build and run locally:

```bash
docker build -t mlops-app .
docker run -d --name mlops-app -p 5000:5000 \
  -e MONGODB_URL="..." \
  -e AWS_ACCESS_KEY_ID="..." \
  -e AWS_SECRET_ACCESS_KEY="..." \
  -e AWS_DEFAULT_REGION="us-east-1" \
  mlops-app
```

## CI/CD Deployment

Pushing to the `main` branch triggers `.github/workflows/aws.yaml`:

1. **Continuous-Integration** (GitHub-hosted) — builds the Docker image and pushes it to Amazon ECR
2. **Continuous-Deployment** (self-hosted EC2 runner) — pulls the image and runs the `mlops-app` container on port **5000**

### GitHub Secrets

Configure these under **Settings → Secrets and variables → Actions**:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `ECR_REPO` | ECR repository name |
| `MONGODB_URL` | MongoDB Atlas connection string |

### EC2 setup

1. Launch an Ubuntu EC2 instance and install Docker
2. Register a **self-hosted GitHub Actions runner** on the instance
3. Install the runner as a systemd service (recommended):

```bash
cd ~/actions-runner
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

4. Open **inbound TCP port 5000** in the instance security group
5. Whitelist the EC2 public IP in **MongoDB Atlas → Network Access**

### Access the deployed app

```
http://<EC2_PUBLIC_IP>:5000
```

Example: `http://13.223.72.203:5000`

> **Note:** `0.0.0.0` is a server bind address — use the EC2 public IP or `localhost` in your browser, not `0.0.0.0`.

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Vehicle insurance prediction form |
| `POST` | `/` | Submit form data and get a prediction |
| `GET` | `/train` | Trigger the full training pipeline |

## AWS Resources

| Resource | Purpose |
|----------|---------|
| **S3 bucket** | Model registry (`MODEL_BUCKET_NAME` in `src/constants/__init__.py`) |
| **ECR repository** | Stores Docker images for deployment |
| **EC2 instance** | Runs the self-hosted runner and production container |
| **IAM user** | Credentials for S3, ECR, and GitHub Actions |


## Useful Commands (EC2)

```bash
docker ps                          # Check running containers
docker logs mlops-app --tail 50    # View application logs
curl http://127.0.0.1:5000/        # Local health check
sudo ./svc.sh status               # Self-hosted runner service status
```

## License

See [LICENSE](LICENSE) for details.
