import fs from "fs";
import path from "path";

const chunkRoot = path.resolve(".next/static/chunks");
const totalBudgetKb = Number(process.env.BUNDLE_TOTAL_BUDGET_KB || 900);
const largestChunkBudgetKb = Number(process.env.BUNDLE_LARGEST_CHUNK_BUDGET_KB || 220);

function getJsFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...getJsFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith(".js")) {
      files.push(fullPath);
    }
  }
  return files;
}

if (!fs.existsSync(chunkRoot)) {
  console.error(`Missing chunk directory: ${chunkRoot}`);
  process.exit(1);
}

const jsFiles = getJsFiles(chunkRoot);
const sized = jsFiles
  .map((file) => ({ file, bytes: fs.statSync(file).size }))
  .sort((a, b) => b.bytes - a.bytes);

const totalBytes = sized.reduce((sum, item) => sum + item.bytes, 0);
const totalKb = totalBytes / 1024;
const largest = sized[0];
const largestKb = largest ? largest.bytes / 1024 : 0;

console.log(`Bundle check: total JS chunks ${totalKb.toFixed(1)}KB (budget ${totalBudgetKb}KB)`);
if (largest) {
  console.log(
    `Bundle check: largest chunk ${path.basename(largest.file)} ${largestKb.toFixed(1)}KB (budget ${largestChunkBudgetKb}KB)`
  );
}

if (totalKb > totalBudgetKb || largestKb > largestChunkBudgetKb) {
  console.error("Bundle budget exceeded.");
  console.error("Top 5 chunks:");
  sized.slice(0, 5).forEach((item) => {
    console.error(`- ${path.basename(item.file)}: ${(item.bytes / 1024).toFixed(1)}KB`);
  });
  process.exit(1);
}
