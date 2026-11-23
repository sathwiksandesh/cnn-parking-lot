import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"--> Starting FastAPI on 0.0.0.0:{port} (Python {sys.version.split()[0]})", flush=True)
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, log_level="info")
