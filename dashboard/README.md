# L.I.F.E THEORY Dashboard

## Local run

- Ensure Python 3.10+ and install: pip install -r ../requirements-streamlit.txt
- Start: streamlit run streamlit_app.py
- Metrics source: ../evidence/latest.json

## Docker

- Build: docker build -f ../Dockerfile.streamlit -t life-streamlit:local ..
- Run: docker run -p 8501:8501 life-streamlit:local

## Azure (App Service for Containers)

- Provision with Terraform in infra/terraform
- Push image via GH Actions (Build and Push Streamlit Container)
- Configure Web App with image and restart (Deploy Streamlit to Azure App Service)
