const express = require("express");
const cors = require("cors");
const multer = require("multer");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 5000;

app.use(cors());
const upload = multer({ dest: "uploads/" });

app.post("/api/predict", upload.single("file"), (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No file uploaded" });

  const uploadedFilePath = path.join(__dirname, req.file.path);
  const pythonProcess = spawn("python", ["predict.py", uploadedFilePath]);

  let output = "";
  let errorOutput = "";

  pythonProcess.stdout.on("data", (data) => {
    output += data.toString();
  });

  pythonProcess.stderr.on("data", (data) => {
    errorOutput += data.toString();
    console.error("PYTHON STDERR:", data.toString());
  });

  pythonProcess.on("close", (code) => {
    fs.unlinkSync(uploadedFilePath); // delete temp file

    if (code !== 0 || errorOutput) {
      return res.status(500).json({ error: "Prediction failed. Check backend logs." });
    }

    try {
      const result = JSON.parse(output);
      return res.json(result);
    } catch (parseErr) {
      console.error("JSON PARSE ERROR:", parseErr.message);
      console.error("Raw Output:", output);
      return res.status(500).json({ error: "Invalid model output format." });
    }
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Backend running at http://localhost:${PORT}`);
});
