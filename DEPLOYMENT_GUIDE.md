# NIST RAG Agent - Deployment Guide

Comprehensive guide for deploying the NIST RAG Agent across local, cloud, and production environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployments](#cloud-deployments)
  - [AWS](#aws-deployment)
  - [Azure](#azure-deployment)
  - [Google Cloud](#google-cloud-deployment)
  - [IBM Cloud / Watson X](#ibm-cloud--watson-x-deployment)
  - [DigitalOcean](#digitalocean-deployment)
- [Production Best Practices](#production-best-practices)
- [Scaling Strategies](#scaling-strategies)
- [Monitoring & Logging](#monitoring--logging)

---

## Local Development

### Standard Python Setup

**Best for**: Development, testing, and small-scale usage

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
export OPENAI_API_KEY="sk-your-key-here"
export USE_HUGGINGFACE="true"          # Use HF dataset (default)
export DATASET_SPLIT="train"            # 424K examples
export TOP_K="5"                        # Retrieval count

# 5. Run agent
python agent.py

# 6. Or start API service
python api_service.py
# Access at http://localhost:8000
```

### Configuration Options

```bash
# Environment Variables
export OPENAI_API_KEY="sk-..."         # Required
export OPENAI_MODEL="gpt-4o"           # LLM model (default: gpt-4o)
export USE_HUGGINGFACE="true"          # Use HF dataset (default: true)
export DATASET_SPLIT="train"           # train (424K) or valid (106K)
export TOP_K="5"                       # Number of retrieved docs
export HF_HOME="/custom/cache"         # HuggingFace cache location
```

### Using Local Embeddings (Offline)

```bash
# Disable HuggingFace dataset
export USE_HUGGINGFACE="false"

# Agent will use local embeddings from embeddings/ directory
python agent.py
```

### Development with Hot Reload

```bash
# Install uvicorn with reload support
pip install uvicorn[standard]

# Run with auto-reload
uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
```

---

## Docker Deployment

### Basic Docker Setup

**Best for**: Containerized local development, reproducible environments

#### Build and Run

```bash
# Build image
docker build -t nist-rag-agent:latest .

# Run container
docker run -d \
  --name nist-rag-agent \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-your-key-here" \
  -e USE_HUGGINGFACE="true" \
  -e DATASET_SPLIT="train" \
  -v $(pwd)/.cache:/app/.cache \
  nist-rag-agent:latest

# Check logs
docker logs -f nist-rag-agent

# Test
curl http://localhost:8000/health
```

#### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Custom Docker Compose Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    image: nist-rag-agent:latest
    restart: always
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=gpt-4o
      - USE_HUGGINGFACE=true
      - DATASET_SPLIT=train
      - TOP_K=5
    volumes:
      - cache-data:/app/.cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 4G

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api

volumes:
  cache-data:
    driver: local
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Cloud Deployments

### AWS Deployment

#### Option 1: EC2 Instance

**Best for**: Full control, custom configurations

```bash
# 1. Launch EC2 instance
# - Instance type: t3.xlarge or larger (4GB+ RAM)
# - AMI: Ubuntu 22.04 LTS
# - Storage: 50GB+ EBS
# - Security Group: Allow ports 22, 80, 443, 8000

# 2. Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.10 python3.10-venv python3-pip docker.io docker-compose

# 4. Clone and setup
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# 5. Configure environment
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
USE_HUGGINGFACE=true
DATASET_SPLIT=train
EOF

# 6. Deploy with Docker
docker-compose up -d

# 7. Setup systemd service for auto-start
sudo tee /etc/systemd/system/nist-rag-agent.service << EOF
[Unit]
Description=NIST RAG Agent
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/nist-rag-agent
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
User=ubuntu

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable nist-rag-agent
sudo systemctl start nist-rag-agent
```

#### Option 2: ECS (Elastic Container Service)

**Best for**: Managed container orchestration, auto-scaling

```bash
# 1. Build and push to ECR
aws ecr create-repository --repository-name nist-rag-agent
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker build -t nist-rag-agent:latest .
docker tag nist-rag-agent:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nist-rag-agent:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nist-rag-agent:latest

# 2. Create ECS task definition (task-definition.json)
```

```json
{
  "family": "nist-rag-agent",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "nist-rag-agent",
      "image": "YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/nist-rag-agent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "USE_HUGGINGFACE",
          "value": "true"
        },
        {
          "name": "DATASET_SPLIT",
          "value": "train"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nist-rag-agent",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

```bash
# 3. Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# 4. Create ECS cluster
aws ecs create-cluster --cluster-name nist-rag-cluster

# 5. Create service
aws ecs create-service \
  --cluster nist-rag-cluster \
  --service-name nist-rag-service \
  --task-definition nist-rag-agent \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=nist-rag-agent,containerPort=8000"
```

#### Option 3: Lambda (Serverless)

**Best for**: Event-driven, pay-per-use scenarios

Due to the size of the HuggingFace dataset (~7GB), Lambda is **not recommended** for this application. Consider ECS or EC2 instead.

---

### Azure Deployment

#### Option 1: Azure VM

**Best for**: Full control, Azure ecosystem integration

```bash
# 1. Create VM
az vm create \
  --resource-group nist-rag-rg \
  --name nist-rag-vm \
  --image Ubuntu2204 \
  --size Standard_D4s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --public-ip-sku Standard

# 2. Open ports
az vm open-port --port 80 --resource-group nist-rag-rg --name nist-rag-vm
az vm open-port --port 443 --resource-group nist-rag-rg --name nist-rag-vm
az vm open-port --port 8000 --resource-group nist-rag-rg --name nist-rag-vm

# 3. SSH and setup (same as EC2 steps 3-7)
ssh azureuser@your-vm-ip
```

#### Option 2: Azure Container Instances (ACI)

**Best for**: Simple container deployments without orchestration

```bash
# 1. Create container registry
az acr create --resource-group nist-rag-rg --name nistragacr --sku Basic

# 2. Build and push image
az acr build --registry nistragacr --image nist-rag-agent:latest .

# 3. Store OpenAI key in Key Vault
az keyvault create --name nist-rag-kv --resource-group nist-rag-rg
az keyvault secret set --vault-name nist-rag-kv --name openai-key --value "sk-your-key"

# 4. Deploy container
az container create \
  --resource-group nist-rag-rg \
  --name nist-rag-agent \
  --image nistragacr.azurecr.io/nist-rag-agent:latest \
  --cpu 2 \
  --memory 8 \
  --registry-login-server nistragacr.azurecr.io \
  --registry-username $(az acr credential show --name nistragacr --query username -o tsv) \
  --registry-password $(az acr credential show --name nistragacr --query passwords[0].value -o tsv) \
  --dns-name-label nist-rag-agent \
  --ports 8000 \
  --environment-variables \
    USE_HUGGINGFACE=true \
    DATASET_SPLIT=train \
  --secure-environment-variables \
    OPENAI_API_KEY=$(az keyvault secret show --vault-name nist-rag-kv --name openai-key --query value -o tsv)

# 5. Get public IP
az container show --resource-group nist-rag-rg --name nist-rag-agent --query ipAddress.fqdn
```

#### Option 3: Azure Kubernetes Service (AKS)

**Best for**: Production, high availability, auto-scaling

```bash
# 1. Create AKS cluster
az aks create \
  --resource-group nist-rag-rg \
  --name nist-rag-aks \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --generate-ssh-keys

# 2. Get credentials
az aks get-credentials --resource-group nist-rag-rg --name nist-rag-aks

# 3. Create Kubernetes resources (see Kubernetes section below)
```

---

### Google Cloud Deployment

#### Option 1: Compute Engine VM

**Best for**: Full control, custom configurations

```bash
# 1. Create VM instance
gcloud compute instances create nist-rag-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server

# 2. Configure firewall
gcloud compute firewall-rules create allow-nist-rag \
  --allow=tcp:8000 \
  --target-tags=http-server

# 3. SSH and setup
gcloud compute ssh nist-rag-vm --zone=us-central1-a
# Follow EC2 setup steps 3-7
```

#### Option 2: Cloud Run

**Best for**: Serverless, auto-scaling, pay-per-use

```bash
# 1. Build and push to Artifact Registry
gcloud artifacts repositories create nist-rag-repo \
  --repository-format=docker \
  --location=us-central1

gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT/nist-rag-repo/nist-rag-agent:latest

# 2. Deploy to Cloud Run
gcloud run deploy nist-rag-agent \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT/nist-rag-repo/nist-rag-agent:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --min-instances 1 \
  --max-instances 10 \
  --set-env-vars USE_HUGGINGFACE=true,DATASET_SPLIT=train \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --allow-unauthenticated

# 3. Get service URL
gcloud run services describe nist-rag-agent --region us-central1 --format 'value(status.url)'
```

**Note**: Cloud Run has an 8GB memory limit. For the full training split, consider using validation split or GKE instead.

#### Option 3: Google Kubernetes Engine (GKE)

**Best for**: Production, high availability

```bash
# 1. Create GKE cluster
gcloud container clusters create nist-rag-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4

# 2. Get credentials
gcloud container clusters get-credentials nist-rag-cluster --zone us-central1-a

# 3. Deploy (see Kubernetes section below)
```

---

### IBM Cloud / Watson X Deployment

#### Option 1: Virtual Server (VPC)

**Best for**: Full control, IBM Cloud ecosystem integration

```bash
# 1. Create VPC and subnet (if not exists)
ibmcloud is vpc-create nist-rag-vpc
ibmcloud is subnet-create nist-rag-subnet nist-rag-vpc --zone us-south-1 --ipv4-cidr-block 10.240.0.0/24

# 2. Create SSH key
ibmcloud is key-create nist-rag-key @~/.ssh/id_rsa.pub

# 3. Create security group
ibmcloud is security-group-create nist-rag-sg nist-rag-vpc
ibmcloud is security-group-rule-add nist-rag-sg inbound tcp --port-min 22 --port-max 22
ibmcloud is security-group-rule-add nist-rag-sg inbound tcp --port-min 80 --port-max 80
ibmcloud is security-group-rule-add nist-rag-sg inbound tcp --port-min 443 --port-max 443
ibmcloud is security-group-rule-add nist-rag-sg inbound tcp --port-min 8000 --port-max 8000

# 4. Create virtual server instance
ibmcloud is instance-create nist-rag-instance \
  nist-rag-vpc \
  us-south-1 \
  bx2-4x16 \
  nist-rag-subnet \
  --image ibm-ubuntu-22-04-minimal-amd64-1 \
  --keys nist-rag-key \
  --security-groups nist-rag-sg

# 5. Get floating IP
ibmcloud is floating-ip-reserve nist-rag-ip --zone us-south-1
ibmcloud is instance-network-interface-floating-ip-add nist-rag-instance primary nist-rag-ip

# 6. SSH and setup
ssh root@your-floating-ip
# Follow standard setup steps
```

#### Option 2: Code Engine (Serverless Containers)

**Best for**: Serverless, auto-scaling, pay-per-use

```bash
# 1. Create Code Engine project
ibmcloud ce project create --name nist-rag-project

# 2. Select project
ibmcloud ce project select --name nist-rag-project

# 3. Build container image
ibmcloud ce build create --name nist-rag-build \
  --source https://github.com/yourusername/nist-rag-agent \
  --commit main \
  --strategy dockerfile \
  --dockerfile Dockerfile

ibmcloud ce buildrun submit --build nist-rag-build --wait

# 4. Create secret for OpenAI key
ibmcloud ce secret create --name openai-secret \
  --from-literal OPENAI_API_KEY=sk-your-key-here

# 5. Deploy application
ibmcloud ce application create --name nist-rag-agent \
  --build-source nist-rag-build \
  --cpu 2 \
  --memory 8G \
  --port 8000 \
  --min-scale 1 \
  --max-scale 10 \
  --env-from-secret openai-secret \
  --env USE_HUGGINGFACE=true \
  --env DATASET_SPLIT=train \
  --timeout 300 \
  --request-timeout 300

# 6. Get application URL
ibmcloud ce application get --name nist-rag-agent --output url
```

**Code Engine Application Configuration**:

```yaml
# app-config.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: nist-rag-agent
  namespace: nist-rag-project
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/target: "100"
    spec:
      containerConcurrency: 10
      timeoutSeconds: 300
      containers:
      - image: us.icr.io/namespace/nist-rag-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: OPENAI_API_KEY
        - name: USE_HUGGINGFACE
          value: "true"
        - name: DATASET_SPLIT
          value: "train"
        resources:
          requests:
            cpu: 2000m
            memory: 8Gi
          limits:
            cpu: 2000m
            memory: 8Gi
```

#### Option 3: Red Hat OpenShift on IBM Cloud

**Best for**: Enterprise Kubernetes, hybrid cloud

```bash
# 1. Create OpenShift cluster
ibmcloud oc cluster create vpc-gen2 \
  --name nist-rag-openshift \
  --zone us-south-1 \
  --version 4.14_openshift \
  --flavor bx2.4x16 \
  --workers 3 \
  --vpc-id nist-rag-vpc \
  --subnet-id nist-rag-subnet

# 2. Get cluster config
ibmcloud oc cluster config --cluster nist-rag-openshift --admin

# 3. Create project
oc new-project nist-rag

# 4. Create secret
oc create secret generic openai-secret \
  --from-literal=OPENAI_API_KEY=sk-your-key-here \
  -n nist-rag

# 5. Deploy application
oc new-app https://github.com/yourusername/nist-rag-agent \
  --name=nist-rag-agent \
  --strategy=docker \
  -e USE_HUGGINGFACE=true \
  -e DATASET_SPLIT=train \
  -n nist-rag

# 6. Set resources
oc set resources deployment/nist-rag-agent \
  --requests=cpu=1,memory=4Gi \
  --limits=cpu=2,memory=8Gi \
  -n nist-rag

# 7. Add secret
oc set env deployment/nist-rag-agent \
  --from=secret/openai-secret \
  -n nist-rag

# 8. Expose service
oc expose service/nist-rag-agent -n nist-rag

# 9. Get route
oc get route nist-rag-agent -n nist-rag
```

#### Option 4: Integration with Watson X.ai

**Best for**: Using IBM Watson X foundation models instead of OpenAI

Modify `agent.py` to support Watson X:

```python
import os
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

class NistRagAgentWatsonX:
    def __init__(self, use_watsonx=False):
        self.use_watsonx = use_watsonx
        
        if self.use_watsonx:
            # Watson X configuration
            self.watsonx_credentials = {
                "url": "https://us-south.ml.cloud.ibm.com",
                "apikey": os.getenv("WATSONX_API_KEY")
            }
            self.watsonx_project_id = os.getenv("WATSONX_PROJECT_ID")
            
            # Initialize Watson X model
            self.model = Model(
                model_id="ibm/granite-13b-chat-v2",
                params={
                    GenParams.DECODING_METHOD: "greedy",
                    GenParams.MAX_NEW_TOKENS: 1000,
                    GenParams.TEMPERATURE: 0.7,
                },
                credentials=self.watsonx_credentials,
                project_id=self.watsonx_project_id
            )
        else:
            # Use OpenAI
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    def query(self, question):
        if self.use_watsonx:
            # Watson X query
            response = self.model.generate_text(prompt=question)
            return response
        else:
            # OpenAI query (existing implementation)
            return self.agent.invoke({
                "input": question,
                "chat_history": self.session_histories.get(session_id, [])
            })
```

**Environment Setup for Watson X**:

```bash
# Install Watson X SDK
pip install ibm-watson-machine-learning>=1.0.335

# Configure environment
export WATSONX_API_KEY="your-watsonx-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export USE_WATSONX="true"

# Deploy with Watson X
python agent.py
```

**Docker Deployment with Watson X**:

```dockerfile
# Dockerfile.watsonx
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ibm-watson-machine-learning>=1.0.335

COPY . .

ENV USE_WATSONX=true

CMD ["python", "api_service.py"]
```

```bash
# Build and run
docker build -f Dockerfile.watsonx -t nist-rag-agent-watsonx:latest .

docker run -d \
  --name nist-rag-watsonx \
  -p 8000:8000 \
  -e WATSONX_API_KEY="your-key" \
  -e WATSONX_PROJECT_ID="your-project-id" \
  -e USE_WATSONX="true" \
  -e USE_HUGGINGFACE="true" \
  -e DATASET_SPLIT="train" \
  nist-rag-agent-watsonx:latest
```

#### Watson X.data Integration

**For storing embeddings and documents in Watson X.data**:

```python
from ibm_watson_data_api import WatsonDataAPI

# Initialize Watson X.data client
watsonx_data = WatsonDataAPI(
    api_key=os.getenv("WATSONX_DATA_API_KEY"),
    service_url="https://us-south.dataplatform.cloud.ibm.com"
)

# Store embeddings
def store_embeddings_watsonx(embeddings, metadata):
    watsonx_data.create_asset(
        asset_type="embedding",
        name="nist-embeddings",
        data=embeddings,
        metadata=metadata
    )

# Retrieve embeddings
def load_embeddings_watsonx():
    asset = watsonx_data.get_asset(
        asset_id="nist-embeddings"
    )
    return asset['data']
```

#### Cost Comparison: Watson X vs OpenAI

| Feature | Watson X | OpenAI GPT-4 |
|---------|----------|--------------|
| Input Tokens | $0.0015/1K | $0.03/1K |
| Output Tokens | $0.002/1K | $0.06/1K |
| Hosting | IBM Cloud | OpenAI |
| Data Residency | Configurable | US-only |
| Enterprise Support | ✅ | Limited |
| On-Premises Option | ✅ | ❌ |

**Benefits of Watson X for Enterprise**:
- **Data sovereignty**: Keep data in specific regions/countries
- **Compliance**: HIPAA, GDPR, FedRAMP certified
- **Cost**: 95% cheaper than OpenAI for high-volume usage
- **Customization**: Fine-tune models on proprietary data
- **Hybrid deployment**: On-premises or cloud
- **No data sharing**: IBM doesn't train on your data

---

### DigitalOcean Deployment

#### Droplet Deployment

**Best for**: Simple, cost-effective cloud hosting

```bash
# 1. Create Droplet via UI or CLI
doctl compute droplet create nist-rag-agent \
  --size s-2vcpu-4gb \
  --image ubuntu-22-04-x64 \
  --region nyc1 \
  --ssh-keys YOUR_SSH_KEY_ID

# 2. Get droplet IP
doctl compute droplet list

# 3. SSH and setup
ssh root@your-droplet-ip
# Follow EC2 setup steps 3-7
```

#### App Platform (PaaS)

**Best for**: Managed platform, automatic scaling

```yaml
# app.yaml
name: nist-rag-agent
region: nyc

services:
  - name: api
    github:
      repo: yourusername/nist-rag-agent
      branch: main
      deploy_on_push: true
    
    build_command: pip install -r requirements.txt
    run_command: uvicorn api_service:app --host 0.0.0.0 --port 8080
    
    environment_slug: python
    instance_size_slug: professional-l
    instance_count: 2
    
    http_port: 8080
    
    health_check:
      http_path: /health
      initial_delay_seconds: 300
      period_seconds: 30
    
    envs:
      - key: OPENAI_API_KEY
        scope: RUN_TIME
        type: SECRET
        value: ${OPENAI_API_KEY}
      - key: USE_HUGGINGFACE
        value: "true"
      - key: DATASET_SPLIT
        value: "train"
```

Deploy:
```bash
doctl apps create --spec app.yaml
```

---

## Kubernetes Deployment

### Complete Kubernetes Setup

**For**: Production deployments on any Kubernetes cluster (AKS, EKS, GKE, self-hosted)

#### 1. Create Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: nist-rag
```

```bash
kubectl apply -f namespace.yaml
```

#### 2. Create Secret for OpenAI Key

```bash
kubectl create secret generic openai-secret \
  --from-literal=OPENAI_API_KEY=sk-your-key-here \
  -n nist-rag
```

#### 3. Create PersistentVolumeClaim for Cache

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nist-rag-cache
  namespace: nist-rag
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: standard
```

```bash
kubectl apply -f pvc.yaml
```

#### 4. Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nist-rag-agent
  namespace: nist-rag
  labels:
    app: nist-rag-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nist-rag-agent
  template:
    metadata:
      labels:
        app: nist-rag-agent
    spec:
      containers:
      - name: nist-rag-agent
        image: your-registry/nist-rag-agent:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: OPENAI_API_KEY
        - name: USE_HUGGINGFACE
          value: "true"
        - name: DATASET_SPLIT
          value: "train"
        - name: TOP_K
          value: "5"
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        volumeMounts:
        - name: cache
          mountPath: /app/.cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 10
      volumes:
      - name: cache
        persistentVolumeClaim:
          claimName: nist-rag-cache
```

```bash
kubectl apply -f deployment.yaml
```

#### 5. Create Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: nist-rag-service
  namespace: nist-rag
spec:
  type: LoadBalancer
  selector:
    app: nist-rag-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
```

```bash
kubectl apply -f service.yaml
```

#### 6. Create Ingress (Optional)

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: nist-rag-ingress
  namespace: nist-rag
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - nist-rag.yourdomain.com
    secretName: nist-rag-tls
  rules:
  - host: nist-rag.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: nist-rag-service
            port:
              number: 80
```

```bash
kubectl apply -f ingress.yaml
```

#### 7. Create HorizontalPodAutoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nist-rag-hpa
  namespace: nist-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nist-rag-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

```bash
kubectl apply -f hpa.yaml
```

---

## Production Best Practices

### 1. Environment Configuration

```bash
# Use environment-specific configs
# .env.production
OPENAI_API_KEY=sk-prod-key
OPENAI_MODEL=gpt-4o
USE_HUGGINGFACE=true
DATASET_SPLIT=train
TOP_K=5
LOG_LEVEL=info

# .env.staging
OPENAI_API_KEY=sk-staging-key
OPENAI_MODEL=gpt-4o-mini
USE_HUGGINGFACE=true
DATASET_SPLIT=valid
TOP_K=3
LOG_LEVEL=debug
```

### 2. Secrets Management

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name nist-rag/openai-key \
  --secret-string "sk-your-key"

# Azure Key Vault
az keyvault secret set \
  --vault-name nist-rag-kv \
  --name openai-key \
  --value "sk-your-key"

# GCP Secret Manager
echo -n "sk-your-key" | gcloud secrets create openai-key --data-file=-

# HashiCorp Vault
vault kv put secret/nist-rag openai_key=sk-your-key
```

### 3. Load Balancing & High Availability

**Nginx Configuration** (`nginx.conf`):

```nginx
upstream nist_rag_backend {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name nist-rag.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name nist-rag.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    client_max_body_size 10M;
    
    location / {
        proxy_pass http://nist_rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 3;
    }
    
    location /health {
        proxy_pass http://nist_rag_backend/health;
        access_log off;
    }
}
```

### 4. Caching Strategy

```python
# Add Redis caching for frequently asked questions
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_response(ttl=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(question, *args, **kwargs):
            cache_key = f"nist_rag:{hash(question)}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Get response
            response = func(question, *args, **kwargs)
            
            # Cache response
            redis_client.setex(
                cache_key,
                ttl,
                json.dumps(response)
            )
            
            return response
        return wrapper
    return decorator
```

### 5. Rate Limiting

```python
# Add rate limiting with Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/query")
@limiter.limit("10/minute")
async def query_agent(request: Request, query_request: QueryRequest):
    # ... existing code
```

---

## Scaling Strategies

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api

  api:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - USE_HUGGINGFACE=true
      - DATASET_SPLIT=train
    volumes:
      - shared-cache:/app/.cache

volumes:
  shared-cache:
```

```bash
# Scale to 5 instances
docker-compose -f docker-compose.scale.yml up -d --scale api=5
```

### Vertical Scaling

```bash
# Increase resources for single instance
docker run -d \
  --name nist-rag-agent \
  --cpus="4" \
  --memory="16g" \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-..." \
  nist-rag-agent:latest
```

### Dataset Optimization

```python
# Use validation split for faster startup
agent = NistRagAgent(
    dataset_split="valid",  # 106K examples vs 424K
    top_k=3                 # Reduce retrieval count
)

# Or use local embeddings for fastest startup
agent = NistRagAgent(
    use_huggingface=False,
    embeddings_dir="./embeddings"
)
```

---

## Monitoring & Logging

### Prometheus Metrics

Add to `api_service.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Metrics
query_counter = Counter('nist_rag_queries_total', 'Total queries')
query_duration = Histogram('nist_rag_query_duration_seconds', 'Query duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.post("/query")
@query_duration.time()
async def query_agent(request: QueryRequest):
    query_counter.inc()
    # ... existing code
```

### ELK Stack Logging

```yaml
# docker-compose.elk.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

  api:
    build: .
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    depends_on:
      - logstash
```

### CloudWatch (AWS)

```python
import boto3
import logging

cloudwatch = boto3.client('cloudwatch')

def log_query_metrics(duration, status):
    cloudwatch.put_metric_data(
        Namespace='NISTRagAgent',
        MetricData=[
            {
                'MetricName': 'QueryDuration',
                'Value': duration,
                'Unit': 'Seconds'
            },
            {
                'MetricName': 'QueryCount',
                'Value': 1,
                'Unit': 'Count'
            }
        ]
    )
```

### Health Check Endpoint

```python
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agent": {
            "model": agent.model,
            "dataset": agent.use_huggingface,
            "sessions": len(agent.session_histories)
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
```

---

## Cost Optimization

### Tips for Reducing Costs

1. **Use validation split** for smaller deployments:
   ```python
   agent = NistRagAgent(dataset_split="valid")  # 106K vs 424K examples
   ```

2. **Implement caching** to reduce OpenAI API calls

3. **Use spot/preemptible instances**:
   ```bash
   # AWS Spot
   aws ec2 run-instances --instance-market-options "MarketType=spot"
   
   # GCP Preemptible
   gcloud compute instances create --preemptible
   ```

4. **Auto-scaling** based on usage:
   - Scale down during off-hours
   - Set minimum instances to 0 for dev environments

5. **Regional deployment** closer to users reduces latency and costs

---

## Backup & Disaster Recovery

### Backup Strategy

```bash
# Backup cache directory (contains FAISS indices)
tar -czf nist-rag-cache-backup-$(date +%Y%m%d).tar.gz .cache/

# Upload to S3
aws s3 cp nist-rag-cache-backup-*.tar.gz s3://your-backup-bucket/

# Or to Azure Blob
az storage blob upload \
  --account-name youraccount \
  --container-name backups \
  --file nist-rag-cache-backup-*.tar.gz
```

### Restore Process

```bash
# Download from backup
aws s3 cp s3://your-backup-bucket/nist-rag-cache-backup-latest.tar.gz .

# Extract
tar -xzf nist-rag-cache-backup-latest.tar.gz

# Restart service
docker-compose restart
```

---

## Security Hardening

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Or use Let's Encrypt (production)
certbot --nginx -d nist-rag.yourdomain.com
```

### API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/query")
async def query_agent(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate token
    if credentials.credentials != os.getenv("API_TOKEN"):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # ... existing code
```

### Network Security

```bash
# Firewall rules (example)
# Only allow HTTPS from specific IPs
sudo ufw allow from 10.0.0.0/8 to any port 443
sudo ufw deny 8000  # Block direct access to API
```

---

## Troubleshooting

### Common Issues

**Issue: Out of memory during first run**
```bash
# Solution: Use validation split or add swap
export DATASET_SPLIT="valid"
# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Issue: Slow API responses**
```bash
# Solution: Increase workers or cache responses
uvicorn api_service:app --workers 4

# Add Redis caching (see caching section)
```

**Issue: Dataset download fails**
```bash
# Solution: Increase timeout, retry, or pre-download
export HF_HUB_DOWNLOAD_TIMEOUT=600
# Or pre-download
huggingface-cli download ethanolivertroy/nist-cybersecurity-training
```

---

## Summary

| Deployment Type | Best For | Setup Time | Cost | Scalability |
|----------------|----------|------------|------|-------------|
| Local Python | Development | 30 min | Free | Low |
| Docker Local | Testing | 35 min | Free | Low |
| EC2/VM | Full control | 45 min | $$ | Medium |
| ECS/ACI | Managed containers | 1 hour | $$$ | High |
| Cloud Run | Serverless | 30 min | $$ | High |
| IBM Code Engine | Serverless | 40 min | $$ | High |
| Watson X | Enterprise AI | 1 hour | $ | High |
| Kubernetes | Production | 2 hours | $$$$ | Very High |

**Recommended for Production**: Kubernetes (AKS/EKS/GKE) with auto-scaling and load balancing.

**Recommended for Quick Start**: Docker Compose on single VM (EC2/Azure VM/GCE/IBM VSI).

**Recommended for Serverless**: Google Cloud Run, Azure Container Instances, or IBM Code Engine.

**Recommended for Enterprise/Compliance**: IBM Watson X with data sovereignty and on-premises options.

**Cost Optimization**: Watson X provides 95% cost savings vs OpenAI for high-volume usage.

---

For questions or issues, see [README.md](README.md) or [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).
