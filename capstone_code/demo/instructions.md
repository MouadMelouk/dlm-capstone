
# Quickstart Guide

## 1. Configure Backend
* Run:
```bash
grep -R --exclude-dir=capstone_frontend "REPLACE THIS" .
````

* Open each highlighted location and replace `REPLACE THIS` with your API keys and/or local paths. Specific instructions are present in each location in the repository.

## 2. Configure Frontend and Storage APIs
* Run:
```bash
grep -R "REPLACE THIS" ./capstone_frontend
```

* Update the placeholders in your frontend code the same way.

## 3. Install Conda Environments

```bash
cd /env_yamls
conda env create -f deepfakebench.yml
conda env create -f vllm.yml
```

## 4. Run Backend

```bash
conda activate vllm
python run_LLM_backend.py
```

> The script will verbose-print database- and LLM-connection statuses to indicate the chatbot is live.

## 5. Start Frontend

```bash
cd capstone_frontend
npm install    # or use npx if preferred
npm run dev
```

```
