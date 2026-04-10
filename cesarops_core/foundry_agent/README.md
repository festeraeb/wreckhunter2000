# Foundry Chat Agent (scaffold)

This folder contains a minimal hosted agent scaffold you can build, push to ACR, and deploy as a Foundry hosted agent.

Files added:

- `app.py` — Flask app with `/chat` endpoint. Uses `MODEL_ENDPOINT` and `MODEL_KEY` env vars to proxy requests to your model. Without those it runs in `echo-mode` for quick testing.
- `requirements.txt` — Python deps.
- `Dockerfile` — container image.
- `.foundry/agent-metadata.yaml` — metadata skeleton for Foundry.

Quick steps
1. Build and tag the image (replace `<youracr>`):

```powershell
docker build -t <youracr>.azurecr.io/foundry-chat-agent:latest .
```

2. Login & push to ACR:

```powershell
az acr login --name <youracr>
docker push <youracr>.azurecr.io/foundry-chat-agent:latest
```

3. Create a hosted agent in Azure AI Foundry (portal) that uses the pushed image. In the agent configuration, set environment variables `MODEL_ENDPOINT` and `MODEL_KEY` to point to your model deployment (Azure OpenAI or other REST endpoint).

4. Test the running container endpoint (replace host/port with your agent endpoint):

```powershell
curl -X POST https://<your-agent-endpoint>/chat -H "Content-Type: application/json" -d '{"input":"hello"}'
```

Notes
- If you don't have an Azure OpenAI model, you can still run the container in echo-mode for 1–2 days to interact via `/chat`.
- If you want, I can generate an automated deploy script once you confirm your ACR name and Foundry project endpoint.
