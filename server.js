// ============================================================
//  server.js  –  Student Mark Extraction API
//  Two features:
//    1. Audio  → Whisper transcribes → DeepSeek extracts marks
//    2. Images → Vision model reads test papers → extracts marks
// ============================================================

require('dotenv').config();

const express = require('express');
const multer  = require('multer');
const cors    = require('cors');
const fs      = require('fs');
const { OpenAI } = require('openai');


// ============================================================
//  SECTION 1: CONFIG
//  All settings live here. Change models/tokens in one place.
// ============================================================

const CONFIG = {
  port         : process.env.PORT || 3001,
  hfToken      : process.env.HF_TOKEN,  // Set in .env file or environment
  whisperUrl   : 'https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3',
  deepSeekModel: 'deepseek-ai/DeepSeek-V4-Flash:fastest',
  visionModel  : 'Qwen/Qwen3.6-35B-A3B:featherless-ai',
};

// Vercel uses a read-only filesystem — uploads must go to /tmp
const IS_VERCEL  = process.env.VERCEL === '1';
const AUDIO_DIR  = IS_VERCEL ? '/tmp/uploads' : 'uploads';
const IMAGES_DIR = IS_VERCEL ? '/tmp/images'  : 'images';


// ============================================================
//  SECTION 2: AI PROMPTS
//  Instructions sent to each AI model.
// ============================================================

// Sent to DeepSeek to extract student marks from transcribed audio
const AUDIO_PROMPT = `
You are a precise data extraction assistant specialized in parsing mixed Bengali/English student mark records.
Your ONLY task is to extract student IDs, marks, and exam types from the input text and return a STRICTLY VALID JSON array.
Do not output any explanations, markdown, code blocks, or extra text.

OUTPUT FORMAT (exact):
[{"student id": "XXX-XX-XXX", "mark": 15, "examtype": "quiz1"}]

PARSING RULES:

1. DIGIT MERGING
   - Raw input may contain digits separated by spaces (e.g., "2 6 2 1 5 5 5 0").
   - FIRST, merge consecutive space-separated digits into a single number.
   - NEVER treat spaced digits as separate values.

2. DYNAMIC ID FORMATTING
   - 8-digit numbers → split as first 3 - next 2 - last 3 → "XXX-XX-XXX"
     Example: 26215550 → "262-15-550"
   - 1-3 digit numbers (short suffix) → left-pad with zeros to 3 digits
     Example: 6 → 006, 241 → 241
   - PREFIX INHERITANCE: Use the "XXX-XX-" prefix from the last full 8-digit ID seen.
     If no full ID seen yet, default prefix is "000-00-".
     Example: After "262-15-550", a short ID "241" becomes "262-15-241"
   - Exception: If text says "section 25", use "232-25-" for short suffixes 001-099.

3. SEQUENTIAL MARK EXTRACTION
   - Keywords that mean a mark follows: "got", "গাট", "marks", "নম্বর"
   - Parse left-to-right in blocks: [ID] [keyword] [mark number]
   - Marks are integers 0-100. Never confuse ID digits with marks.

4. EXAM TYPE
   - Look for: quiz1, quiz2, midterm, final, assignment, lab, viva
   - If found → apply to ALL records. If not found → default to "quiz1"

5. OUTPUT RULES
   - One JSON object per valid ID + mark pair
   - "mark" must be a number (not a string)
   - Only three keys allowed: "student id", "mark", "examtype"
   - Empty or unparseable input → return exactly: []
   - Output ONLY raw JSON starting with [ and ending with ]. No markdown, no backticks.

EXAMPLES:
Input:  23215380 গাট 13 820 গাট 15 895 গাট 9
Output: [{"student id":"232-15-380","mark":13,"examtype":"quiz1"},{"student id":"232-15-820","mark":15,"examtype":"quiz1"},{"student id":"232-15-895","mark":9,"examtype":"quiz1"}]

Input:  105 গাট 70 208 got 92 midterm
Output: [{"student id":"232-15-105","mark":70,"examtype":"midterm"},{"student id":"232-15-208","mark":92,"examtype":"midterm"}]

Input:  2 6 2 1 5 5 5 0 got 14 2 4 1 got 15
Output: [{"student id":"262-15-550","mark":14,"examtype":"quiz1"},{"student id":"262-15-241","mark":15,"examtype":"quiz1"}]
`.trim();

// Sent to the vision model to extract data from scanned test paper images
const IMAGE_PROMPT = `
You are an expert OCR and document analysis AI. Extract Student ID and Exam Mark from the provided test paper image.

The document will match ONE of these two formats:

FORMAT A (Simple Class Test):
- Student ID: labeled "Student ID" in the header
- Mark: a prominent number in the top area (often circled)

FORMAT B (Daffodil International University Exam):
- Student ID: labeled "ID Number" near the top left
- Mark: the "Total" row's "Marks Obtained" value in the marks table

RULES:
1. Identify the format first, then extract accordingly.
2. Copy the Student ID exactly as written (keep hyphens, e.g. "232-15-241").
3. Extract the mark as a plain number (0-100).

Return ONLY this raw JSON — no markdown, no explanation:
{"studentId": "extracted_id_here", "mark": extracted_mark_number}
`.trim();


// ============================================================
//  SECTION 3: APP SETUP
//  Initialize Express, middleware, upload folders, AI client.
// ============================================================

const app = express();
app.use(cors());
app.use(express.json());

// Create upload folders if they don't exist yet
[AUDIO_DIR, IMAGES_DIR].forEach(dir => {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// AI client — points to HuggingFace's OpenAI-compatible API
const aiClient = new OpenAI({
  baseURL: 'https://router.huggingface.co/v1',
  apiKey : CONFIG.hfToken,
});


// ============================================================
//  SECTION 4: FILE UPLOAD CONFIG (Multer)
//  Controls how uploaded files are saved to disk.
// ============================================================

// Save uploaded files with a timestamp prefix to avoid name collisions
function createDiskStorage(folder) {
  return multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, folder + '/'),
    filename   : (_req,  file, cb) => cb(null, Date.now() + '-' + file.originalname),
  });
}

// Audio uploads — any audio format, max 50 MB
const audioUpload = multer({
  storage   : createDiskStorage(AUDIO_DIR),
  limits    : { fileSize: 50 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('audio/')) return cb(null, true);
    cb(new Error('Unsupported audio format: ' + file.mimetype));
  },
});

// Image uploads — jpg/png/webp/gif only, max 10 MB, up to 10 files at once
const imageUpload = multer({
  storage   : createDiskStorage(IMAGES_DIR),
  limits    : { fileSize: 10 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
    if (allowed.includes(file.mimetype)) return cb(null, true);
    cb(new Error('Unsupported image format: ' + file.mimetype));
  },
});


// ============================================================
//  SECTION 5: AUDIO PROCESSING FUNCTIONS
//  Three steps: transcribe → extract → parse
// ============================================================

// Step A: Send audio file to Whisper → returns transcript text
async function transcribeAudio(filePath, mimeType) {
  const controller = new AbortController();
  const timeout    = setTimeout(() => controller.abort(), 30_000); // 30 second timeout

  const response = await fetch(CONFIG.whisperUrl, {
    method : 'POST',
    headers: { 'Authorization': `Bearer ${CONFIG.hfToken}`, 'Content-Type': mimeType },
    body   : fs.readFileSync(filePath),
    signal : controller.signal,
  });

  clearTimeout(timeout);

  if (!response.ok) {
    throw new Error(`Whisper error ${response.status}: ${await response.text()}`);
  }

  const data = await response.json();
  return data.text;
}

// Step B: Send transcript text to DeepSeek → returns raw JSON string
async function extractMarksWithDeepSeek(transcriptText) {
  const response = await aiClient.chat.completions.create({
    model      : CONFIG.deepSeekModel,
    max_tokens : 500,
    temperature: 0.1,
    messages: [
      { role: 'system', content: AUDIO_PROMPT },
      { role: 'user',   content: `Extract student marks from this text: "${transcriptText}"` },
    ],
  });
  return response.choices[0].message.content;
}

// Step C: Parse DeepSeek's raw JSON string into a clean array
function parseDeepSeekOutput(rawJson, transcriptText) {
  // Remove any markdown code fences the model may have added
  const cleaned = rawJson.replace(/```(?:json)?|```/g, '').trim();

  // Pull out just the JSON array part
  const jsonString = cleaned.match(/\[[\s\S]*\]/)?.[0] ?? cleaned;
  const parsed     = JSON.parse(jsonString);

  // Normalize one record — handles different key naming styles models may use
  function normalize(item) {
    return {
      'student id'  : item['student id'] || item.studentId || item.student_id || 'N/A',
      mark          : typeof item.mark === 'number' ? item.mark : (parseInt(item.mark) || 0),
      examtype      : item.examtype || item.examType || 'quiz1',
      transcription : transcriptText,
    };
  }

  if (Array.isArray(parsed))      return parsed.map(normalize);
  if (typeof parsed === 'object') return [normalize(parsed)];

  throw new Error('Unexpected format in DeepSeek response');
}

// Fallback: extract marks with regex when DeepSeek is unavailable
function extractMarksWithRegex(text) {
  const results = [];
  const seen    = new Set(); // track IDs already added

  // Try full 8-digit or hyphenated IDs first (e.g. 23215380 or 232-15-380)
  const fullIdPattern = /(\d{3}-?\d{2}-?\d{3})\D+?(\d{1,3})(?:\s*(quiz\d*|exam\d*|test\d*|final|midterm))?/gi;
  for (const match of text.matchAll(fullIdPattern)) {
    const digits = match[1].replace(/-/g, '');
    const id = digits.length === 8
      ? `${digits.slice(0,3)}-${digits.slice(3,5)}-${digits.slice(5)}`
      : match[1];

    if (!seen.has(id)) {
      seen.add(id);
      results.push({
        'student id'  : id,
        mark          : parseInt(match[2]) || 0,
        examtype      : (match[3] || 'quiz1').toLowerCase(),
        transcription : text,
      });
    }
  }

  // If no full IDs found, try short 3-digit IDs (e.g. 380, 820)
  if (results.length === 0) {
    const shortIdPattern = /(?:id\s*)?(\d{3})\D+?(?:got\s*)?(\d{1,3})(?:\s*(quiz\d*|exam\d*|test\d*|final|midterm))?/gi;
    for (const match of text.matchAll(shortIdPattern)) {
      const id = `232-15-${match[1]}`;
      if (!seen.has(id)) {
        seen.add(id);
        results.push({
          'student id'  : id,
          mark          : parseInt(match[2]) || 0,
          examtype      : (match[3] || 'quiz1').toLowerCase(),
          transcription : text,
        });
      }
    }
  }

  // If nothing matched at all, return a helpful error record
  if (results.length === 0) {
    return [{ 'student id': 'N/A', mark: 0, examtype: 'N/A', transcription: text, message: 'Could not extract student data' }];
  }

  return results;
}


// ============================================================
//  SECTION 6: IMAGE PROCESSING FUNCTIONS
//  Two steps: send image to AI → parse the response
// ============================================================

// Step A: Send image to vision model → returns raw text response
async function analyzeImage(file) {
  // Convert image file to base64 so it can be sent over the API
  const base64  = fs.readFileSync(file.path).toString('base64');
  const dataUrl = `data:${file.mimetype};base64,${base64}`;

  const response = await aiClient.chat.completions.create({
    model     : CONFIG.visionModel,
    max_tokens: 1000,
    messages  : [{
      role   : 'user',
      content: [
        { type: 'text',      text      : IMAGE_PROMPT },
        { type: 'image_url', image_url : { url: dataUrl } },
      ],
    }],
  });

  return response.choices[0].message.content;
}

// Step B: Parse the vision model's text response into a clean array
function parseImageOutput(rawText) {
  let cleanJson = rawText.trim();

  // If the model wrapped output in markdown code fences, strip them
  const fenceMatch = rawText.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    cleanJson = fenceMatch[1].trim();
  }

  // If there's extra text around the JSON, pull out just the object
  const objectMatch = cleanJson.match(/\{[\s\S]*\}/);
  if (objectMatch) {
    cleanJson = objectMatch[0];
  }

  const parsed = JSON.parse(cleanJson);
  const items  = Array.isArray(parsed) ? parsed : [parsed];

  // Normalize field names — models may use different naming styles
  return items.map(item => ({
    studentId: item.studentId || item.student_id || item.studentid || item['student id'] || 'N/A',
    mark      : typeof item.mark === 'number' ? item.mark : (parseInt(item.mark) || 0),
  }));
}


// ============================================================
//  SECTION 7: UTILITY
// ============================================================

// Silently delete a file after it has been processed
function deleteFile(filePath) {
  if (filePath && fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
}


// ============================================================
//  SECTION 8: API ROUTES
// ============================================================

// Health check — quick way to confirm the server is alive
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok' });
});

// ── POST /api/analyze-audio ──────────────────────────────────
// 1. Receive audio file
// 2. Transcribe with Whisper
// 3. Extract marks with DeepSeek (regex as fallback)
// 4. Return array of { student id, mark, examtype, transcription }

app.post('/api/analyze-audio', audioUpload.single('audio'), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No audio file provided' });

  try {
    // Transcribe audio to text
    console.log(`[Audio] Transcribing: ${file.filename}`);
    const transcript = await transcribeAudio(file.path, file.mimetype);
    console.log(`[Audio] Transcript: ${transcript}`);

    // Extract marks from transcript
    let results;
    try {
      console.log('[Audio] Extracting marks with DeepSeek...');
      const rawOutput = await extractMarksWithDeepSeek(transcript);
      results = parseDeepSeekOutput(rawOutput, transcript);
      console.log('[Audio] DeepSeek succeeded.');
    } catch (err) {
      console.warn('[Audio] DeepSeek failed — switching to regex fallback:', err.message);
      results = extractMarksWithRegex(transcript);
    }

    deleteFile(file.path);
    console.log('[Audio] Final results:', results);
    res.json(results);

  } catch (err) {
    deleteFile(file?.path);
    console.error('[Audio] Fatal error:', err.message);
    res.status(500).json({ error: 'Failed to analyze audio', details: err.message });
  }
});

// ── POST /api/analyze-images ─────────────────────────────────
// 1. Receive up to 10 image files
// 2. For each image: send to vision model → parse response
// 3. Return array of { studentId, mark }

app.post('/api/analyze-images', imageUpload.array('images', 10), async (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No images provided' });
  }

  console.log(`[Image] Processing ${files.length} image(s)...`);
  const allResults = [];

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    console.log(`[Image] Analyzing ${i + 1}/${files.length}: ${file.filename}`);

    try {
      const rawOutput = await analyzeImage(file);
      console.log(`[Image] Raw AI output:\n---\n${rawOutput}\n---`);

      const parsed = parseImageOutput(rawOutput);
      console.log(`[Image] Parsed:`, parsed);
      allResults.push(...parsed);

    } catch (err) {
      console.error(`[Image] Failed for ${file.filename}:`, err.message);
      allResults.push({ studentId: 'N/A', mark: 0, error: err.message });
    }
  }

  res.json(allResults);
});


// ============================================================
//  SECTION 9: GLOBAL ERROR HANDLER
//  Catches errors from multer (e.g. file too large).
// ============================================================

app.use((err, _req, res, _next) => {
  const message = err.code === 'LIMIT_FILE_SIZE'
    ? 'File too large (max 50 MB for audio, 10 MB for images)'
    : err.message;
  res.status(500).json({ error: message });
});


// ============================================================
//  SECTION 10: START SERVER
//  Runs locally in dev. Exports for Vercel serverless in prod.
// ============================================================

if (!IS_VERCEL) {
  app.listen(CONFIG.port, () => {
    console.log(`Server running on http://localhost:${CONFIG.port}`);
  });
}

// Required for Vercel to treat this file as a serverless function
module.exports = app;