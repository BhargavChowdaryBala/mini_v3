# Ultimate Free Microservices Deployment Guide 🚀

This guide will walk you step-by-step through deploying your entire ANPR application across three free services (HuggingFace, Render, and Vercel). By splitting the system, we avoid memory limits and ensure stable, 24/7 uptime for free.

All required code files have been generated for you in the folder: `C:\Users\bharg\Desktop\for_n\microservices_deployment\`

---

## Architecture Overview

1.  **ML Inference Service (HuggingFace Spaces)**: Runs YOLOv8 and PaddleOCR. Highly compute-intensive, requires minimal setup.
2.  **Backend API (Render.com)**: A lightweight Flask API that talks to MongoDB and forwards images to HuggingFace.
3.  **Frontend (Vercel)**: Your React/Vite UI.

---

## Step 1: Deploy ML Inference Service (HuggingFace Spaces)

HuggingFace Spaces allows you to run Python ML models for free.

1.  Go to [huggingface.co/spaces](https://huggingface.co/spaces) and log in.
2.  Click **Create new Space**.
    *   **Space name**: `anpr-inference` (or whatever you like)
    *   **License**: `mit`
    *   **Select the Space SDK**: Choose **Docker** -> **Blank** (this gives maximum freedom, though you can also just choose Gradio/Streamlit if you prefer, but we are using a raw Flask API).
    *   *Wait, actually, since we are using Flask, the easiest Space SDK is simply* **Gradio** (even though we aren't using the UI, Gradio natively runs Python scripts and exposes port 7860!).
    *   **Recommended**: Choose **Gradio**.
3.  Click **Create Space**.
4.  In the File browser for your Space (or via clone), upload these files from `microservices_deployment/ml_service/`:
    *   `app.py`
    *   `requirements.txt`
    *   **`best.pt`** (Copy this from your existing project folder)
5.  Wait for the Space to Build.
6.  Once it says **Running**, grab the Direct URL to your API. It will look something like this:
    `https://yourusername-anpr-inference.hf.space/detect`
    *(Save this URL for Step 3!)*

---

## Step 2: Set up Database (MongoDB Atlas)

If you don't already have a cloud MongoDB database:

1.  Go to [mongodb.com/atlas](https://www.mongodb.com/atlas) and create a free tier cluster.
2.  Under **Database Access**, create a user with a strong password.
3.  Under **Network Access**, click "Add IP Address" and select **Allow Access from Anywhere (0.0.0.0/0)**.
4.  Click **Connect** -> **Drivers** and copy your Connection String. It looks like:
    `mongodb+srv://username:password@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority`
    *(Save this string for Step 3, and ensure you replace `<password>` with the actual password!)*

---

## Step 3: Deploy Backend API (Render)

Render provides free web services for standard APIs.

1.  Push the code from `microservices_deployment/backend/` to a new GitHub repository named `anpr-backend-api`.
2.  Go to [dashboard.render.com](https://dashboard.render.com/), sign in, and click **New +** -> **Web Service**.
3.  Connect your GitHub account and select your `anpr-backend-api` repository.
4.  Configure the service:
    *   **Name**: `anpr-api`
    *   **Language**: `Python`
    *   **Branch**: `main`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn main:app --timeout 120` *(We use gunicorn for stability, and a high timeout since ML inference takes time)*.
    *   **Instance Type**: `Free`
5.  Scroll down to **Environment Variables** and add two keys:
    *   Key: `MONGO_URI`
        Value: *(Paste your connection string from Step 2)*
    *   Key: `ML_SERVICE_URL`
        Value: *(Paste your HuggingFace URL from Step 1)*
6.  Click **Create Web Service**. Wait 5-10 minutes for it to build and run.
7.  Copy your Render URL (e.g., `https://anpr-api.onrender.com`).

---

## Step 4: Deploy Frontend UI (Vercel)

Finally, deploy your React/Vite site.

1.  Push your frontend folder to a GitHub repository called `anpr-frontend`.
2.  Go to [vercel.com](https://vercel.com/) and click **Add New** -> **Project**.
3.  Import the `anpr-frontend` repository.
4.  In the deployment settings, locate **Environment Variables**:
    *   Key: `VITE_BACKEND_URL`
    *   Value: *(Paste your Render URL from Step 3 without an ending slash, e.g., `https://anpr-api.onrender.com`)*
5.  Click **Deploy**.

---

## 🛠️ Troubleshooting

**"The frontend says Fetch Failed or gives a CORS error!"**
Ensure that the `VITE_BACKEND_URL` exactly matches the running Render URL, and that it doesn't end with a `/`.

**"The backend logs say 'ML Service failed or timed out'"**
Go to your HuggingFace Space. Does it say "Sleeping"? Free Spaces go to sleep if inactive for 48 hours. Just load the Space page in your browser to wake it up. Also ensure your `ML_SERVICE_URL` environment variable properly ends with `/detect`.

**"MongoDB Connection Timeout"**
Go to MongoDB Atlas Network Access and ensure `0.0.0.0/0` is active.

🎉 **You're Done! Your industrial-scale ANPR system is completely free and practically crash-proof!**
