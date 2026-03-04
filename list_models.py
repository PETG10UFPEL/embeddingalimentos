import os
from google import genai

key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("Faltou a variável GOOGLE_API_KEY (ou GEMINI_API_KEY).")

client = genai.Client(api_key=key)

print("Modelos que suportam generateContent:\n")
for m in client.models.list():
    methods = getattr(m, "supported_generation_methods", None) or getattr(m, "supportedGenerationMethods", None) or []
    name = getattr(m, "name", "")
    if any(str(x).lower().endswith("generatecontent") or str(x) == "generateContent" for x in methods):
        print(f"- {name} | baseModelId={getattr(m,'base_model_id',None) or getattr(m,'baseModelId',None)} | methods={methods}")
