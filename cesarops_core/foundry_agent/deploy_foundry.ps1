param(
    [string]$AcrName = $(Read-Host "ACR name (without .azurecr.io)"),
    [string]$ImageTag = $(Read-Host "Image tag (default: latest)" -Default "latest"),
    [string]$ProjectEndpoint = $(Read-Host "Foundry project endpoint URL")
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

if (-not $AcrName) { Write-Error "ACR name is required."; exit 1 }
if (-not $ProjectEndpoint) { Write-Error "Project endpoint is required."; exit 1 }

$image = "$AcrName.azurecr.io/foundry-chat-agent:$ImageTag"

Write-Host "Building image $image..."
docker build -t $image .
if ($LASTEXITCODE -ne 0) { Write-Error "docker build failed"; exit 1 }

Write-Host "Logging into ACR $AcrName..."
az acr login --name $AcrName
if ($LASTEXITCODE -ne 0) { Write-Error "az acr login failed"; exit 1 }

Write-Host "Pushing image to ACR..."
docker push $image
if ($LASTEXITCODE -ne 0) { Write-Error "docker push failed"; exit 1 }

# Update .foundry/agent-metadata.yaml
$metaPath = Join-Path $ScriptDir ".foundry\agent-metadata.yaml"
if (-Not (Test-Path $metaPath)) { Write-Error "Missing $metaPath"; exit 1 }

$content = Get-Content $metaPath -Raw
$content = $content -replace 'projectEndpoint: ".*"', "projectEndpoint: \"$ProjectEndpoint\""
$content = $content -replace 'azureContainerRegistry: ".*"', "azureContainerRegistry: \"$AcrName.azurecr.io\""
$content = $content -replace 'image: ".*"', "image: \"$image\""

Set-Content -Path $metaPath -Value $content -Encoding UTF8
Write-Host "Updated $metaPath with project endpoint, ACR and image tag."

Write-Host "\nNext steps:"
Write-Host "1) In Azure AI Foundry, create or update a hosted agent and point its container image to: $image"
Write-Host "2) Configure environment variables for the agent: MODEL_ENDPOINT (your model REST URL) and MODEL_KEY (API key)"
Write-Host "3) Start the agent and test the /chat endpoint (agent will expose the container HTTP endpoint)." 
Write-Host "\nExample test (replace <agent-host> with your agent hostname):"
Write-Host "curl -X POST https://<agent-host>/chat -H \"Content-Type: application/json\" -d '{\"input\":\"hello\"}'"
