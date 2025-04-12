const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { FaissStore } = require("@langchain/community/vectorstores/faiss");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { TaskType } = require("@google/generative-ai");
const { readFileSync, existsSync, rmSync } = require('fs');
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY || "AIzaSyAN-gWTcAgcNXPr4pFHdNNsUW437IswlBc";
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "text-embedding-004" });
const llmModel = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

const PERSONA_PROMPT = `You are Jenas Anton's virtual assistant. Your role is to introduce Jenas to recruiters or answer any professional inquiries in a clear, approachable, and friendly manner...`;

const app = express();
app.use(bodyParser.json());
app.use(cors());

class GoogleEmbeddings {
  async embedDocuments(texts) {
    console.log("Embedding documents...");
    const result = await embeddingModel.batchEmbedContents({
      requests: texts.map(text => ({
        content: { parts: [{ text }] },
        taskType: TaskType.RETRIEVAL_DOCUMENT,
      })),
    });
    return result.embeddings.map(e => e.values);
  }

  async embedQuery(text) {
    const result = await embeddingModel.embedContent({
      content: { parts: [{ text }] },
      taskType: TaskType.RETRIEVAL_QUERY,
    });
    return result.embedding.values;
  }
}

const txtPath = path.resolve(__dirname, 'assistant_profile.txt');
const faissDirectory = path.resolve(__dirname, 'faiss_index');
const FORCE_REBUILD = false; // set to true to always rebuild

function chunkText(text, size = 2) {
  const sentences = text.split(/(?<=\.)\s+/);
  const chunks = [];
  for (let i = 0; i < sentences.length; i += size) {
    chunks.push(sentences.slice(i, i + size).join(" "));
  }
  return chunks;
}

const loadDocTexts = () => {
  const fileContent = readFileSync(txtPath, 'utf-8');
  return fileContent
    .split('\n\n')
    .flatMap(paragraph => chunkText(paragraph, 2))
    .filter(line => line.trim() !== '');
};

let vectorStore = null;

async function initializeVectorStore() {
  console.log("Initializing vector store...");
  const docTexts = loadDocTexts();
  console.log(`Loaded ${docTexts.length} document chunks`);

  try {
    const embeddings = new GoogleEmbeddings();

    if (FORCE_REBUILD || !existsSync(faissDirectory)) {
      if (FORCE_REBUILD && existsSync(faissDirectory)) {
        rmSync(faissDirectory, { recursive: true });
        console.log("Deleted old FAISS index for rebuild.");
      }

      console.log("Creating new FAISS index...");
      vectorStore = await FaissStore.fromTexts(
        docTexts,
        docTexts.map((_, i) => ({ id: i.toString() })),
        embeddings
      );
      await vectorStore.save(faissDirectory);
    } else {
      console.log("Loading existing FAISS index...");
      vectorStore = await FaissStore.load(faissDirectory, embeddings);
    }
  } catch (error) {
    console.error("Error initializing vector store:", error);
    throw error;
  }
}

initializeVectorStore().catch(error => {
  console.error("Failed to initialize vector store:", error);
  process.exit(1);
});

async function generateAnswer(question, relevantTexts) {
  const context = relevantTexts.map((text, i) => `${i + 1}. ${text}`).join("\n");

  const extraContext = `
Jenas' Profile:
- GitHub: https://github.com/Jenas-Anton
- LinkedIn: https://www.linkedin.com/in/jenas-anton
`;

  const prompt = `${PERSONA_PROMPT}\n\nQuestion: ${question}\n\nContext:\n${context}\n${extraContext}\n\nAnswer:`;

  try {
    const result = await llmModel.generateContent({
      contents: [{ parts: [{ text: prompt }] }]
    });

    const response = await result.response;
    return response.text().trim();
  } catch (error) {
    console.error("Error generating answer:", error);
    return "Sorry, I encountered an error while processing your request.";
  }
}

app.post('/ask', async (req, res) => {
  const { question, k = 5 } = req.body;
  console.log("Received question:", question);

  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  try {
    if (!vectorStore) {
      return res.status(503).json({ error: 'Vector store not yet initialized' });
    }

    const results = await vectorStore.similaritySearch(question, k);
    const relevantTexts = results.map(result => result.pageContent);

    console.log("Top relevant chunks:");
    console.log(relevantTexts);

    if (!relevantTexts.length) {
      return res.json({ answer: "I don't have enough information to answer that about Jenas." });
    }

    const answer = await generateAnswer(question, relevantTexts);
    res.json({ answer });
  } catch (error) {
    console.error("Error processing request:", error);
    res.status(500).json({ error: 'Internal server error', message: error.message });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: "ok" });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

module.exports = app; // Export the app if necessary for testing or future expansion
