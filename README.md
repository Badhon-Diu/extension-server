# Student Mark Extraction API

An AI-powered API that extracts student marks from **voice recordings** and **scanned test paper images**.

---

## How It Works

```
Audio File  →  Whisper (transcribe)  →  DeepSeek (extract marks)  →  JSON Response
Image File  →  Vision AI (read paper)                              →  JSON Response
```

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Environment Setup](#environment-setup)
3. [Local Development](#local-development)
4. [API Reference](#api-reference)
5. [Testing with Postman](#testing-with-postman)
6. [Deployment to Vercel](#deployment-to-vercel)
7. [Troubleshooting](#troubleshooting)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Express.js** | Web server framework |
| **Multer** | Handles file uploads |
| **Whisper** (HuggingFace) | Transcribes audio to text |
| **DeepSeek** (HuggingFace) | Extracts marks from transcribed text |
| **Qwen Vision** (HuggingFace) | Reads and extracts marks from images |
| **Vercel** | Serverless cloud deployment |

---

## Environment Setup

### Step 1 — Get a HuggingFace Token

1. Go to [huggingface.co](https://huggingface.co) and create a free account
2. Go to **Settings → Access Tokens**
3. Click **New Token**, give it a name, and copy it

### Step 2 — Create a `.env` file

In your project's server folder, create a file named `.env` and add:

```env
HF_TOKEN=your_huggingface_token_here
```

> ⚠️ **Never commit `.env` to GitHub.** It is already listed in `.gitignore`.

---

## Local Development

```bash
# 1. Install all dependencies
npm install

# 2. Start the development server (auto-restarts on file changes)
npm run dev

# 3. Or start the production server
npm start
```

The server will run at: **`http://localhost:3001`**

---

## API Reference

### `GET /api/health`

Checks if the server is running.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /api/analyze-audio`

Uploads an audio recording and returns extracted student marks.

**Request details:**

| Field | Value |
|-------|-------|
| Method | `POST` |
| Content-Type | `multipart/form-data` |
| Field name | `audio` |
| Max file size | `50 MB` |
| Supported formats | `webm`, `wav`, `mp3`, `mp4`, `ogg`, `m4a` |

**Example response:**
```json
[
  {
    "student id": "232-15-045",
    "mark": 8,
    "examtype": "quiz1",
    "transcription": "45 got 8"
  },
  {
    "student id": "232-15-380",
    "mark": 13,
    "examtype": "quiz1",
    "transcription": "380 got 13"
  }
]
```

**Response field descriptions:**

| Field | Description |
|-------|-------------|
| `student id` | Student ID in `XXX-XX-XXX` format |
| `mark` | Numeric mark (0–100) |
| `examtype` | Exam type (e.g. `quiz1`, `midterm`, `final`) |
| `transcription` | Raw text that Whisper extracted from the audio |

---

### `POST /api/analyze-images`

Uploads up to 10 scanned test paper images and returns extracted student IDs and marks.

**Request details:**

| Field | Value |
|-------|-------|
| Method | `POST` |
| Content-Type | `multipart/form-data` |
| Field name | `images` |
| Max files | `10` |
| Max size per file | `10 MB` |
| Supported formats | `jpg`, `jpeg`, `png`, `webp`, `gif` |

**Example response:**
```json
[
  {
    "studentId": "232-15-045",
    "mark": 5
  },
  {
    "studentId": "232-15-046",
    "mark": 8
  }
]
```

**Response field descriptions:**

| Field | Description |
|-------|-------------|
| `studentId` | Student ID read directly from the paper |
| `mark` | Numeric mark extracted from the paper |

---

## Testing with Postman

### Health Check

| Setting | Value |
|---------|-------|
| Method | `GET` |
| URL | `http://localhost:3001/api/health` |

Expected response: `{ "status": "ok" }`

---

### Audio Analysis

**Step-by-step setup:**

1. Open Postman and create a new request
2. Set **Method** to `POST`
3. Set **URL** to `http://localhost:3001/api/analyze-audio`
4. Click the **Body** tab → select **form-data**
5. Add a key-value row:
   - **Key:** `audio`
   - **Type:** change the dropdown from `Text` to **`File`**
   - **Value:** click **Select Files** and choose your audio file

```
Key      Type    Value
------   ------  -------------------
audio    File    your_recording.mp3
```

---

### Image Analysis

**Step-by-step setup:**

1. Open Postman and create a new request
2. Set **Method** to `POST`
3. Set **URL** to `http://localhost:3001/api/analyze-images`
4. Click the **Body** tab → select **form-data**
5. Add one row **per image** — all rows must use the **same key name** `images`:

```
Key      Type    Value
------   ------  -------------------
images   File    test_paper_1.jpg
images   File    test_paper_2.jpg
images   File    test_paper_3.png
```

> ⚠️ **Important:** Use the exact key name `images` for every file. Using different key names will cause an error.

---

## Deployment to Vercel

### Step 1 — Push your code to GitHub

```bash
git add .
git commit -m "initial commit"
git push
```

### Step 2 — Import project in Vercel

1. Go to [vercel.com](https://vercel.com) and log in
2. Click **Add New Project**
3. Select your GitHub repository and click **Import**

### Step 3 — Add environment variable

1. In your Vercel project, go to **Settings → Environment Variables**
2. Add the following:

| Name | Value |
|------|-------|
| `HF_TOKEN` | your HuggingFace token |

### Step 4 — Deploy

Click **Deploy**. Vercel will build and host your API automatically.

After deployment, your API will be available at:
```
https://your-project-name.vercel.app/api/health
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `"Unexpected field"` | Wrong key name in form-data | Use exactly `audio` or `images` (lowercase, no spaces) |
| `"Unsupported format"` | Wrong file type uploaded | Audio: `webm/wav/mp3/mp4/ogg/m4a` — Images: `jpg/png/webp/gif` |
| `"File too large"` | File exceeds size limit | Audio max: 50 MB — Images max: 10 MB each |
| `"Could not extract student data"` | AI couldn't find a valid pattern | Check audio clarity or image quality |
| `500 Internal Server Error` | Usually a missing or invalid `HF_TOKEN` | Check your `.env` file or Vercel environment variables |
| Server doesn't start | Missing dependencies | Run `npm install` first |

---

## License

MIT
