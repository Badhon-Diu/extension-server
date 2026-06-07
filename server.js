// ============================================================
//  server.js  –  Student Mark Extraction API  (v5 – final)
//
//  Features:
//    1. Audio  → Whisper transcribes → DeepSeek-V4-Flash extracts marks
//    2. Images → Vision model reads test papers → extracts marks
//
//  Changes in v5:
//    ✓ Model upgraded: DeepSeek-R1 → DeepSeek-V4-Flash:novita
//      (13B activated params — much faster, fewer 504s)
//    ✓ stripThinkingBlock() now called in parseDeepSeekOutput()
//      (V4-Flash can think — strip <think> before JSON parse)
//    ✓ DeepSeek timeout increased: 20s → 35s
//    ✓ Regex fallback fixed: merges spaced digits before matching
//      e.g. "2 3 2 1 5 3 8 0" → "23215380" before regex runs
//
//  FIX v5.1:
//    ✓ extractMarksWithDeepSeek() — removed stream:true
//      HF router does not support proper async streaming.
//      stream:false returns response.choices[0].message.content directly.
//      Previously fullContent was always "" → assertNotEmpty threw →
//      regex fallback ran → wrong results.
//
//  Response format (both APIs):
//    Success → [{"student id": "XXX-XX-XXX", "mark": 15}, ...]
//    Error   → {"error": "message"}
// ============================================================

require('dotenv').config();

const express = require('express');
const multer  = require('multer');
const cors    = require('cors');
const fs      = require('fs');
const crypto  = require('crypto');
const { OpenAI } = require('openai');


// ============================================================
//  SECTION 1: CONFIG
// ============================================================

const CONFIG = {
  port         : process.env.PORT || 3001,
  hfToken      : process.env.HF_TOKEN,

  // Whisper: best open-source speech-to-text model on HuggingFace
  whisperUrl   : 'https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3',

  // DeepSeek-V4-Flash: 13B activated params (MoE), fast, smart enough for JSON extraction
  // Much faster than R1 (671B) — fewer 504 timeouts on HF router via Novita
  deepSeekModel: 'deepseek-ai/DeepSeek-V4-Flash:novita',

  // Vision model: Gemma 4 31B — real VLM for reading test paper images
  visionModel  : 'google/gemma-4-31B-it:novita',

  // Max images processed in parallel per batch.
  // Keep at 3 — sending more causes 429 concurrency errors on Novita
  imageBatchSize: 3,

  // Timeout limits for each external AI service (milliseconds)
  timeouts: {
    whisper  : 30_000,   // 30s — audio transcription can be slow
    vision   : 40_000,   // 40s — vision model needs time for image analysis
  },
};

// Vercel serverless has a read-only filesystem — audio temp files must go to /tmp
const IS_VERCEL = process.env.VERCEL === '1';
const AUDIO_DIR = IS_VERCEL ? '/tmp/uploads' : 'uploads';


// ============================================================
//  SECTION 2: AI PROMPTS
// ============================================================

// Sent to DeepSeek-V4-Flash to parse a Bengali/English audio transcript into student marks
const AUDIO_PROMPT = `
You are a precise data extraction assistant specialized in parsing mixed Bengali/English student mark records.
Your ONLY task is to extract student IDs and marks from the input text and return a STRICTLY VALID JSON array.
Do not output any explanations, markdown, code blocks, or extra text.

OUTPUT FORMAT (exact):
[{"student id": "XXX-XX-XXX", "mark": 15}]

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

4. OUTPUT RULES
   - One JSON object per valid ID + mark pair
   - Only two keys allowed: "student id" and "mark"
   - "mark" must be a number (not a string)
   - Empty or unparseable input → return exactly: []
   - Output ONLY raw JSON starting with [ and ending with ]. No markdown, no backticks.

EXAMPLES:
Input:  23215380 গাট 13 820 গাট 15 895 গাট 9
Output: [{"student id":"232-15-380","mark":13},{"student id":"232-15-820","mark":15},{"student id":"232-15-895","mark":9}]

Input:  105 গাট 70 208 got 92 midterm
Output: [{"student id":"232-15-105","mark":70},{"student id":"232-15-208","mark":92}]

Input:  2 6 2 1 5 5 5 0 got 14 2 4 1 got 15
Output: [{"student id":"262-15-550","mark":14},{"student id":"262-15-241","mark":15}]

/no_think
`.trim();

// Sent to the vision model to extract student ID and mark from a test paper image.
const IMAGE_PROMPT = `
You are an OCR extraction tool. Look at this test paper image and extract exactly two values.

WHAT TO FIND:

1. Student ID
   Look for a field labeled any of: "Student ID", "ID Number", "ID No", "Roll No"
   Copy the value exactly as written, including hyphens (e.g. "232-15-241").

2. Obtained Mark / Score
   Look for the final awarded score. It may appear as:
   - A circled or boxed number at the top of the paper
   - The value in the "Total" row under the "Marks Obtained" column in a marks table
   - A number next to "Total Marks", "Score", or "Obtained"
   Extract it as a plain integer only (e.g. write 17, not "17/20").

STRICT OUTPUT RULES:
- After your thinking, output ONLY this exact JSON object. Nothing else. No explanation. No markdown. No backticks.
- Format: {"student id": "value here", "mark": number here}
- Example: {"student id": "232-15-290", "mark": 17}
- If a value cannot be found, use null for that field.
- The JSON must start with { and end with }
`.trim();


// ============================================================
//  SECTION 3: APP & MIDDLEWARE SETUP
// ============================================================

const app = express();
app.use(cors());
app.use(express.json());

// Create audio upload folder if it does not exist
if (!fs.existsSync(AUDIO_DIR)) {
  fs.mkdirSync(AUDIO_DIR, { recursive: true });
}

// Single AI client pointing to HuggingFace's OpenAI-compatible API router
const aiClient = new OpenAI({
  baseURL: 'https://router.huggingface.co/v1',
  apiKey : CONFIG.hfToken,
});


// ============================================================
//  SECTION 4: FILE UPLOAD CONFIG (Multer)
// ============================================================

// Audio: saved to disk — Whisper requires reading the file as binary
const audioUpload = multer({
  storage: multer.diskStorage({
    destination: (_req, _file, cb) => cb(null, AUDIO_DIR + '/'),
    filename   : (_req,  file, cb) => cb(null, Date.now() + '-' + file.originalname),
  }),
  limits    : { fileSize: 50 * 1024 * 1024 }, // 50 MB max
  fileFilter: (_req, file, cb) => {
    if (file.mimetype.startsWith('audio/')) return cb(null, true);
    cb(new Error('Unsupported audio format: ' + file.mimetype));
  },
});

// Images: stored in RAM as file.buffer — faster, no disk cleanup needed
const imageUpload = multer({
  storage: multer.memoryStorage(),
  limits : { fileSize: 10 * 1024 * 1024 }, // 10 MB max per image
  fileFilter: (_req, file, cb) => {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/gif'];
    if (allowed.includes(file.mimetype)) return cb(null, true);
    cb(new Error('Unsupported image format: ' + file.mimetype));
  },
});


// ============================================================
//  SECTION 5: IMAGE RESPONSE CACHE
// ============================================================

const imageCache = new Map();

function getCacheKey(buffer) {
  return crypto.createHash('md5').update(buffer).digest('hex');
}


// ============================================================
//  SECTION 6: UTILITY HELPERS
// ============================================================

// Safely delete a file — won't throw if already gone
function deleteFile(filePath) {
  if (filePath && fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
}

// AbortController that auto-cancels after `ms` milliseconds
function createTimeout(ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return {
    signal: controller.signal,
    clear : () => clearTimeout(timer),
  };
}

// Safely convert any mark value to an integer
function normalizeMark(value) {
  if (typeof value === 'number' && !isNaN(value)) return value;
  if (value === null || value === undefined)       return 0;
  const parsed = parseInt(String(value), 10);
  return isNaN(parsed) ? 0 : parsed;
}

// Strip <think>...</think> reasoning blocks
function stripThinkingBlock(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
}


// ============================================================
//  SECTION 7: AUDIO PIPELINE  (simple & clean)
// ============================================================

// Step A: Transcribe audio → Whisper
async function transcribeAudio(filePath, mimeType) {
  const response = await fetch(CONFIG.whisperUrl, {
    method : 'POST',
    headers: {
      'Authorization': `Bearer ${CONFIG.hfToken}`,
      'Content-Type' : mimeType,
    },
    body: fs.readFileSync(filePath),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Whisper error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return data.text;
}

// Step B: Send transcript → DeepSeek-V4-Flash → raw JSON string
async function extractMarksWithDeepSeek(transcriptText) {
  const response = await aiClient.chat.completions.create({
    model   : CONFIG.deepSeekModel,
    stream  : false,
    messages: [
      { role: 'system', content: AUDIO_PROMPT },
      { role: 'user',   content: `Extract student marks from this text: "${transcriptText}"` },
    ],
  });

  const content = response.choices[0]?.message?.content || '';
  console.log('[Audio] DeepSeek raw response:', content);
  return content;
}

// Step C: Parse DeepSeek's raw output → normalized array
function parseDeepSeekOutput(rawJson) {
  // Strip <think>...</think> — V4-Flash may think before answering
  let cleaned = stripThinkingBlock(rawJson);

  // Strip markdown fences
  cleaned = cleaned.replace(/```(?:json)?|```/g, '').trim();

  // Extract JSON array
  const jsonString = cleaned.match(/\[[\s\S]*\]/)?.[0] ?? cleaned;
  const parsed = JSON.parse(jsonString);

  function normalizeRecord(item) {
    return {
      'student id': item['student id'] || item.studentId || item.student_id || 'N/A',
      mark        : normalizeMark(item.mark),
    };
  }

  if (Array.isArray(parsed))      return parsed.map(normalizeRecord);
  if (typeof parsed === 'object') return [normalizeRecord(parsed)];

  throw new Error('Unexpected format in DeepSeek response');
}


// ============================================================
//  SECTION 8: IMAGE PIPELINE  (unchanged)
// ============================================================

// Step A: Send one image to the vision model → returns raw text response
async function analyzeImage(file) {
  const cacheKey = getCacheKey(file.buffer);

  if (imageCache.has(cacheKey)) {
    console.log(`[Image] Cache hit: ${file.originalname}`);
    return imageCache.get(cacheKey);
  }

  const base64  = file.buffer.toString('base64');
  const dataUrl = `data:${file.mimetype};base64,${base64}`;

  const { signal, clear } = createTimeout(CONFIG.timeouts.vision);

  try {
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
      signal,
    });

    const rawOutput = response.choices[0].message.content;
    if (!rawOutput || rawOutput.trim() === '') {
      throw new Error(`Vision model (${CONFIG.visionModel}) returned an empty response`);
    }

    imageCache.set(cacheKey, rawOutput);
    return rawOutput;

  } finally {
    clear();
  }
}

// Step B: Parse the vision model's response into a clean normalized record
function parseImageOutput(rawText) {
  if (!rawText || rawText.trim() === '') {
    throw new Error('Vision model output was empty');
  }

  // Strip <think>...</think> block (Gemma / Qwen think before answering)
  let cleanJson = stripThinkingBlock(rawText);

  // Strip markdown code fences
  const fenceMatch = cleanJson.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    cleanJson = fenceMatch[1].trim();
  }

  // Extract just the JSON object
  const objectMatch = cleanJson.match(/\{[\s\S]*\}/);
  if (objectMatch) {
    cleanJson = objectMatch[0];
  }

  const parsed = JSON.parse(cleanJson);
  const items  = Array.isArray(parsed) ? parsed : [parsed];

  return items.map(item => ({
    'student id': item['student id'] || item.studentId || item.student_id || item.studentid || 'N/A',
    mark        : normalizeMark(item.mark),
  }));
}


// ============================================================
//  SECTION 9: API ROUTES
// ============================================================

// GET /api/health
app.get('/api/health', (_req, res) => {
  res.json({
    status      : 'ok',
    timestamp   : new Date().toISOString(),
    visionModel : CONFIG.visionModel,
    audioModel  : CONFIG.deepSeekModel,
    batchSize   : CONFIG.imageBatchSize,
  });
});

// POST /api/analyze-audio
// Pipeline: audio → Whisper → transcript → DeepSeek → JSON marks
// Response: [{"student id": "XXX-XX-XXX", "mark": 15}] or {"error": "..."}
app.post('/api/analyze-audio', audioUpload.single('audio'), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No audio file provided' });

  console.log(`[Audio] Received: ${file.filename}`);

  const filePath = file.path;

  try {
    console.log('[Audio] Sending to Whisper...');
    const transcript = await transcribeAudio(filePath, file.mimetype);
    console.log(`[Audio] Transcript: "${transcript}"`);

    console.log('[Audio] Sending transcript to DeepSeek-V4-Flash...');
    const rawOutput = await extractMarksWithDeepSeek(transcript);

    const results = parseDeepSeekOutput(rawOutput);
    console.log('[Audio] Final results:', results);

    res.status(200).json(results);

  } catch (err) {
    console.error('[Audio] Error:', err.message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to analyze audio', details: err.message });
    }
  } finally {
    deleteFile(filePath);
  }
});

// POST /api/analyze-images
// Pipeline: Vision model reads each image → extracts student id + mark
// Batched in groups of CONFIG.imageBatchSize (3) to avoid 429 errors
// Response: [{"student id": "XXX-XX-XXX", "mark": 15}] or {"error": "..."}
app.post('/api/analyze-images', imageUpload.array('images', 10), async (req, res) => {
  const files = req.files;
  if (!files || files.length === 0) {
    return res.status(400).json({ error: 'No images provided' });
  }

  const batchSize    = CONFIG.imageBatchSize;
  const totalBatches = Math.ceil(files.length / batchSize);
  console.log(`[Image] Received ${files.length} image(s). Processing in ${totalBatches} batch(es) of ${batchSize}.`);

  const allResults = [];

  for (let i = 0; i < files.length; i += batchSize) {
    const batch       = files.slice(i, i + batchSize);
    const batchNumber = Math.floor(i / batchSize) + 1;

    console.log(`[Image] Batch ${batchNumber}/${totalBatches} — processing ${batch.length} image(s)...`);

    const batchResults = await Promise.all(
      batch.map(async (file, batchIndex) => {
        const globalIndex = i + batchIndex + 1;
        console.log(`[Image] Analyzing ${globalIndex}/${files.length}: ${file.originalname}`);

        try {
          const rawOutput = await analyzeImage(file);
          console.log(`[Image] Raw output for ${file.originalname}:\n${rawOutput}`);

          const parsed = parseImageOutput(rawOutput);
          console.log(`[Image] Parsed for ${file.originalname}:`, parsed);
          return parsed;

        } catch (err) {
          console.error(`[Image] Failed for ${file.originalname}: ${err.message}`);
          return [{ 'student id': 'N/A', mark: 0 }];
        }
      })
    );

    allResults.push(...batchResults.flat());
  }

  console.log(`[Image] All done. Total records extracted: ${allResults.length}`);
  return res.json(allResults);
});


// ============================================================
//  SECTION 10: GLOBAL ERROR HANDLER
// ============================================================

app.use((err, _req, res, _next) => {
  const message = err.code === 'LIMIT_FILE_SIZE'
    ? 'File too large (max 50 MB for audio, 10 MB per image)'
    : err.message || 'An unexpected error occurred';

  console.error('[Server] Unhandled error:', message);
  res.status(400).json({ error: message });
});


// ============================================================
//  SECTION 11: START SERVER
// ============================================================

if (!IS_VERCEL) {
  app.listen(CONFIG.port, () => {
    console.log(`✓ Server running at http://localhost:${CONFIG.port}`);
    console.log(`✓ Vision model : ${CONFIG.visionModel}`);
    console.log(`✓ Audio model  : ${CONFIG.deepSeekModel}`);
    console.log(`✓ Image batch  : ${CONFIG.imageBatchSize} per batch`);
  });
}

module.exports = app;
