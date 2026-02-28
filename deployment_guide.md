# Deployment Guide

This guide explains how to deploy your application to **Render** (Backend) and **Vercel** (Frontend).

## Prerequisites
- A [GitHub](https://github.com/) account.
- A [Render](https://render.com/) account.
- A [Vercel](https://vercel.com/) account.
- A [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) account (for a cloud database).

---

## Step 1: Initialize Git and Push to GitHub

1. Open a terminal in the root directory (`for_n`).
2. Initialize Git:
   ```bash
   git init
   git add .
   git commit -m "Prepare for deployment"
   ```
3. Create a new repository on GitHub and push your code:
   ```bash
   git remote add origin YOUR_GITHUB_REPO_URL
   git branch -M main
   git push -u origin main
   ```

---

## Step 2: Deploy Backend to Render

1. Log in to **Render** and click **New > Web Service**.
2. Connect your GitHub repository.
3. Configure the service:
   - **Name**: `bus-monitoring-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `gunicorn main:app --chdir backend --bind 0.0.0.0:$PORT`
4. Add **Environment Variables**:
   - `MONGO_URI`: Your MongoDB Atlas connection string.
   - `PORT`: 10000 (usually set automatically by Render).
   - `MALLOC_ARENA_MAX`: `2` (Highly recommended for 512MB RAM).

> [!TIP]
> **Video Uploads**: Render has a temporary file system. Uploaded videos will work but will be cleaned up automatically by the system and my built-in cleanup logic.

---

## Step 3: Deploy Frontend to Vercel

1. Log in to **Vercel** and click **Add New > Project**.
2. Import your GitHub repository.
3. Configure the project:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Add **Environment Variables**:
   - `VITE_API_BASE`: The URL of your Render backend (e.g., `https://bus-monitoring-backend.onrender.com`).

---

## Step 4: Verification
Once both are deployed, open your Vercel URL. The frontend should connect to the Render backend and show the live feed and logs.
