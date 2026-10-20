# Coach UI

Human-friendly youth feedback screen (not the Swagger `/docs` page).

## Open

From the repo root (with venv active if you use one):

```bash
python -m http.server 8765
```

Then open [http://127.0.0.1:8765/coach-ui/](http://127.0.0.1:8765/coach-ui/).

Click **一键看演示** for the plant-foot correction sample. Optionally start the API
(`bash scripts/run_analyze_api.sh` or uvicorn on port 8000) and use **用本机 API 再分析一次**.

`/docs` is developer API reference only.
